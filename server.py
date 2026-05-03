#!/usr/bin/env python3
#DEMO CHANGE
"""
TransBuddy Bus Face Verification Server — v12.0.0
Marwadi University

NEW IN v12:
  • Multi-bus support — 115 buses, each identified by bus_id
  • Bus & driver info fetched from DB (bus_detail + driver_detail tables)
  • Live stream frames received from each Pi and served per bus
  • /stream_frame   — Pi pushes JPEG frames for live viewing
  • /live_frame_bus — Dashboard polls per-bus live frame
  • /register_bus   — Pi registers at startup, loads bus/driver info
  • /buses          — Lists all active buses with info + stats
  • All detection records tagged with bus_id
  • Offline replay support (captures taken when Pi was offline)

DECISION TREE (per detected face):
  A. No photo match          -> /not_uni_student
  B. Photo matched, no DB    -> /invalid_alerts (not_in_db)
  C. In DB, no bus sub       -> /invalid_alerts (no_bus_policy)
  D. Has bus, fee UNPAID     -> /unpaid_students
  E. Has bus, fee PAID       -> /valid_students  (ACCESS GRANTED)
"""

import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import copy, os, base64, time, logging, threading, queue
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from flask import Flask, request, jsonify, send_from_directory, Response
from insightface.app import FaceAnalysis
try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


# ── LOGGING ───────────────────────────────────────────────────────────────────
_LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
_fh = logging.FileHandler("server.log", encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter(_LOG_FMT))
_sh = logging.StreamHandler(stream=sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter(_LOG_FMT))
logger = logging.getLogger("transbuddy")
logger.setLevel(logging.INFO)
logger.addHandler(_fh)
logger.addHandler(_sh)
logging.getLogger().setLevel(logging.WARNING)


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    CONFIDENCE_THRESHOLD = 0.45
    MARGIN_THRESHOLD     = 0.03
    FACE_MIN_SIDE_PX     = 36
    FACE_MIN_AREA_RATIO  = 0.004
    FACE_BLUR_THRESHOLD  = 55.0
    FACE_MIN_DET_SCORE   = 0.45

    PHOTO_DATASET_REPO = "Krushang1911/Face_data/photos"
    PHOTO_DATASET_REPO_TYPE = "dataset"
    PHOTO_DATASET_CACHE = Path(".hf_cache") / "face-database"
    DIR_WITH_BUS    = "captures/with_bus"
    DIR_WITHOUT_BUS = "captures/without_bus"
    DIR_INVALID     = "captures/invalid_captures"
    DIR_NOT_UNI     = "captures/not_uni_student"

    # ── Aiven Cloud MySQL ─────────────────────────────────────────────────
    DB_HOST      = "transbuddy-db-1-transbuddy.e.aivencloud.com"
    DB_PORT      = 20742
    DB_USER      = "avnadmin"
    DB_PASSWORD  = "AVNS_IxUzga3f6XjSmzEv6Ej"
    DB_NAME      = "defaultdb"
    DB_SSL_CA    = Path(__file__).resolve().with_name("ca.pem")  # optional: place Aiven CA cert here
    DB_POOL_SIZE = 5   # keep lower for cloud DB (connection limit)

    BUS_CHECK_MODE    = "fee"
    STUDENT_CACHE_TTL = 300

    INSIGHT_CTX = -1
    DET_SIZE    = (320, 320)

    RESULT_HOLD_SECS  = 8
    VALID_STORE_MAX   = 1000
    UNPAID_STORE_MAX  = 1000
    INVALID_STORE_MAX = 1000
    NOT_UNI_STORE_MAX = 500

    # Per-bus stream frame: drop frames older than this from memory
    STREAM_FRAME_TTL  = 15      # seconds

    THROTTLE_SECS  = 3.0
    LIVE_JPEG_Q    = 55
    CAPTURE_JPEG_Q = 88
    ENROLL_JPEG_Q  = 80

    PROOF_DIR         = "proof_images"
    PROOF_RETAIN_DAYS = 30

    # Bus considered inactive after this many seconds without a stream frame
    BUS_INACTIVE_SECS = 60


for _d in [Config.DIR_WITH_BUS, Config.DIR_WITHOUT_BUS,
           Config.DIR_INVALID, Config.DIR_NOT_UNI,
           Config.PROOF_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)


# =============================================================================
# FLASK
# =============================================================================
app = Flask(__name__, static_folder="static", template_folder="templates")


@app.before_request
def _opts():
    if request.method == "OPTIONS":
        r = app.make_default_options_response()
        r.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,HEAD",
            "Access-Control-Allow-Headers": "*",
        })
        return r


@app.after_request
def _cors(resp):
    resp.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,HEAD",
        "Access-Control-Allow-Headers": "*",
        "Cache-Control": "no-cache,no-store,must-revalidate",
    })
    return resp


# =============================================================================
# GLOBAL STATE
# =============================================================================
face_app        = None
embedding_store = {}
enrollment_imgs = {}
emb_lock        = threading.RLock()

# ── Multi-bus registry ─────────────────────────────────────────────────────
# bus_id (str) -> {
#   bus_info: {bus_no, capacity, bus_type, is_active, bus_register_no, ...},
#   driver_info: {driver_name, phone_no, license_no, ...},
#   stream_b64: str,      # latest MJPEG frame
#   stream_ts: str,       # ISO timestamp
#   last_seen: float,     # time.time() of last activity
#   gps: {...},           # latest GPS from upload
#   session_stats: {valid, unpaid, invalid, not_uni},
# }
_bus_registry = {}
_bus_reg_lock = threading.RLock()

# Legacy single-bus state (kept for backward compat)
_IDLE_SCAN = {
    "is_idle": True, "multi_face": False, "face_count": 0, "results": [],
    "summary": {"valid_with_bus": 0, "unpaid": 0, "invalid": 0, "not_uni": 0, "cooldown": 0},
    "location": None, "timestamp": None, "captured_b64": None,
}
latest_scan   = copy.deepcopy(_IDLE_SCAN)
scan_lock     = threading.Lock()
_last_scan_ts = 0.0

_bus_location = {
    "gps_lat": None, "gps_lon": None, "stop_name": None,
    "stop_lat": None, "stop_lon": None, "stop_city": None,
    "stop_state": None, "stop_country": None, "stop_display": None,
    "image_quality": None, "updated_at": None,
}
_bus_loc_lock = threading.Lock()

# Route stores
valid_store   = [];  valid_lock   = threading.Lock()
unpaid_store  = {};  unpaid_lock  = threading.Lock()
invalid_store = [];  inv_lock     = threading.Lock()
not_uni_store = [];  nu_lock      = threading.Lock()

# Cooldown
student_cooldown = {}
cd_lock          = threading.Lock()

_live_frame     = None
_live_frame_b64 = None
_live_lock      = threading.Lock()

_PRIO = {"valid_with_bus": 4, "valid_without_bus": 3,
         "invalid_database": 2, "invalid_person": 1, "on_cooldown": 0}

_throttle       = {};  _throttle_lock  = threading.Lock()
_nodb_reported  = {};  _nodb_lock      = threading.Lock()
_stu_cache      = {};  _stu_cache_lock = threading.Lock()
_upload_sem     = threading.Semaphore(10)  # Allow up to 10 concurrent uploads
_save_q         = queue.Queue(maxsize=200)

# SSE clients
_sse_clients      = []
_sse_clients_lock = threading.Lock()


def _sse_broadcast(event: str, data: str):
    msg = "event: " + str(event) + "\ndata: " + str(data) + "\n\n"
    with _sse_clients_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


# =============================================================================
# HELPERS
# =============================================================================
def _ms(t0):    return f"{(time.time()-t0)*1000:.0f}ms"
def _today():   return datetime.now().strftime("%Y-%m-%d")
def _get_slot():return "morning" if datetime.now().hour < 14 else "evening"


def _status_label(status, category=None):
    if status == "valid_with_bus":
        return "GRANTED"
    if status == "valid_without_bus":
        return "UNPAID"
    if status == "invalid_person":
        return "NO BUS"
    if status == "invalid_database":
        return "NOT IN DB"
    if status == "not_uni":
        return "UNKNOWN"
    if status == "on_cooldown":
        return "COOLDOWN"
    if status == "low_quality" or category == "low_quality":
        return "LOW QUALITY"
    return (str(status or "").replace("_", " ").upper()) or "UNKNOWN"


def _make_label(name, status, category=None):
    base = str(name or "Unknown Person").strip() or "Unknown Person"
    return f"{base} · {_status_label(status, category)}"


def _face_quality(face, frame_bgr, crop_bgr):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    face_w = max(0, x2 - x1)
    face_h = max(0, y2 - y1)
    area_ratio = (face_w * face_h) / float(max(1, w * h))
    blur_score = 0.0
    if crop_bgr is not None and crop_bgr.size > 0:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    det_score = float(getattr(face, "det_score", 0.0) or 0.0)
    reasons = []
    if min(face_w, face_h) < Config.FACE_MIN_SIDE_PX:
        reasons.append("too small")
    if area_ratio < Config.FACE_MIN_AREA_RATIO:
        reasons.append("tiny face")
    if blur_score < Config.FACE_BLUR_THRESHOLD:
        reasons.append("blurred")
    if det_score and det_score < Config.FACE_MIN_DET_SCORE:
        reasons.append("weak detection")
    return {
        "ok": not reasons,
        "reason": ", ".join(reasons),
        "blur_score": round(blur_score, 2),
        "area_ratio": round(area_ratio, 4),
        "det_score": round(det_score, 4) if det_score else None,
    }


def _annotate_face_focus(frame, face_box, kps=None, label=None, color=(255, 255, 255)):
    x1, y1, x2, y2 = [int(v) for v in face_box]
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    bw = max(2, int(min(w, h) * 0.0025))
    corner = max(12, int(min(x2 - x1, y2 - y1) * 0.18))
    col = tuple(int(c) for c in color)

    # Focus-style corner brackets.
    cv2.line(frame, (x1, y1), (min(w - 1, x1 + corner), y1), col, bw)
    cv2.line(frame, (x1, y1), (x1, min(h - 1, y1 + corner)), col, bw)
    cv2.line(frame, (x2, y1), (max(0, x2 - corner), y1), col, bw)
    cv2.line(frame, (x2, y1), (x2, min(h - 1, y1 + corner)), col, bw)
    cv2.line(frame, (x1, y2), (min(w - 1, x1 + corner), y2), col, bw)
    cv2.line(frame, (x1, y2), (x1, max(0, y2 - corner)), col, bw)
    cv2.line(frame, (x2, y2), (max(0, x2 - corner), y2), col, bw)
    cv2.line(frame, (x2, y2), (x2, max(0, y2 - corner)), col, bw)

    if kps is not None:
        for pt in np.asarray(kps):
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px, py), max(2, bw + 1), (255, 255, 255), -1, lineType=cv2.LINE_AA)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        pad_x, pad_y = 8, 6
        tx1 = x1
        ty1 = max(0, y1 - th - baseline - pad_y * 2 - 4)
        tx2 = min(w - 1, tx1 + tw + pad_x * 2)
        ty2 = min(h - 1, ty1 + th + baseline + pad_y * 2)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (10, 12, 18), -1)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), col, 1)
        cv2.putText(frame, label, (tx1 + pad_x, ty2 - pad_y - baseline), font, scale, col, thick, cv2.LINE_AA)


def _annotate_scan_frame(frame, results):
    if frame is None:
        return None
    out = frame.copy()
    palette = {
        "valid_with_bus": (34, 197, 94),
        "valid_without_bus": (245, 158, 11),
        "invalid_person": (239, 68, 68),
        "invalid_database": (168, 85, 247),
        "not_uni": (14, 165, 233),
        "low_quality": (148, 163, 184),
        "on_cooldown": (148, 163, 184),
    }
    for r in results or []:
        bbox = r.get("bbox")
        if not bbox:
            continue
        color = palette.get(r.get("status"), (255, 255, 255))
        label = r.get("label") or _make_label(r.get("display_name") or r.get("name"), r.get("status"), r.get("category"))
        _annotate_face_focus(out, bbox, r.get("kps"), label, color=color)
    return out


def _parse_location(src):
    def _f(k):
        v = src.get(k) or ""
        return str(v).strip() or None
    def _ff(k):
        try:
            v = str(src.get(k) or "").strip()
            return float(v) if v else None
        except:
            return None
    return {
        "gps_lat": _ff("gps_lat"), "gps_lon": _ff("gps_lon"),
        "stop_name": _f("stop_name"), "stop_lat": _ff("stop_lat"),
        "stop_lon": _ff("stop_lon"), "stop_city": _f("stop_city"),
        "stop_state": _f("stop_state"), "stop_country": _f("stop_country"),
        "stop_display": _f("stop_display"), "image_quality": _ff("image_quality"),
        "captured_at": _f("captured_at"), "pickup_id": _f("pickup_id"),
    }


def _update_bus_location(loc):
    with _bus_loc_lock:
        _bus_location.update(loc)
        _bus_location["updated_at"] = datetime.now().isoformat()


# =============================================================================
# DATABASE
# =============================================================================
_db_pool = None


def _db_connect_kwargs():
    kwargs = {
        "host":             Config.DB_HOST,
        "port":             Config.DB_PORT,
        "user":             Config.DB_USER,
        "password":         Config.DB_PASSWORD,
        "database":         Config.DB_NAME,
        "connect_timeout":  10,   # slightly longer for cloud latency
        "autocommit":       True,
        # Aiven requires SSL — always enable even without a local ca.pem
        "ssl_disabled":     False,
    }
    # If you downloaded the Aiven CA cert and placed it as ca.pem next to server.py,
    # use it for strict certificate verification; otherwise TLS still works via
    # the system CA bundle installed in the container (ca-certificates package).
    if Config.DB_SSL_CA.is_file():
        kwargs["ssl_ca"] = str(Config.DB_SSL_CA)
    return kwargs


def _init_db_pool():
    global _db_pool
    try:
        _db_pool = MySQLConnectionPool(
            pool_name="transbuddy", pool_size=Config.DB_POOL_SIZE,
            **_db_connect_kwargs(),
        )
        logger.info(f"DB pool ready | {Config.DB_HOST}/{Config.DB_NAME}")
        return True
    except Exception as e:
        logger.error(f"DB pool failed: {e}")
        return False


def _get_db():
    if _db_pool:
        try:
            return _db_pool.get_connection()
        except:
            pass
    return mysql.connector.connect(**_db_connect_kwargs())


def test_db():
    try:
        c = _get_db()
        cur = c.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) as n FROM students_detail")
        n = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) as n FROM students_detail WHERE fee_status='Paid'")
        paid = cur.fetchone()["n"]
        cur.close(); c.close()
        logger.info(f"DB OK | students={n} fee_paid={paid}")
        return True
    except Exception as e:
        logger.error(f"DB test failed: {e}")
        return False


def fetch_student(gr_no: str):
    gr_no = str(gr_no).strip()
    now = time.time()
    with _stu_cache_lock:
        if gr_no in _stu_cache:
            row, ts = _stu_cache[gr_no]
            if now - ts < Config.STUDENT_CACHE_TTL:
                return row
    row = None
    try:
        c = _get_db()
        cur = c.cursor(dictionary=True)
        cur.execute(
            "SELECT gr_no, enrollment_no, student_name, department, "
            "semester, shift, fee_status, pickup_id "
            "FROM students_detail WHERE gr_no = %s LIMIT 1", (gr_no,)
        )
        row = cur.fetchone()
        cur.close(); c.close()
    except Exception as e:
        logger.error(f"DB fetch gr={gr_no}: {e}")
    with _stu_cache_lock:
        _stu_cache[gr_no] = (row, now)
    return row


def invalidate_student(gr_no):
    with _stu_cache_lock:
        _stu_cache.pop(str(gr_no).strip(), None)


def fetch_bus_and_driver(bus_id: str) -> dict:
    """Fetch bus + driver info from DB. Returns merged dict."""
    bus_id = str(bus_id).strip()
    result = {"bus_id": bus_id, "bus_info": {}, "driver_info": {}}
    try:
        c = _get_db()
        cur = c.cursor(dictionary=True)

        def _norm_row(row: dict) -> dict:
            if not row:
                return {}
            out = {k: (str(v) if v is not None else "") for k, v in row.items()}
            # Normalize common aliases so frontend always has expected keys.
            if not out.get("driver_name") and out.get("name"):
                out["driver_name"] = out["name"]
            if not out.get("driver_name") and out.get("username"):
                out["driver_name"] = out["username"]
            if not out.get("phone_no"):
                out["phone_no"] = out.get("mobile_no") or out.get("phone") or out.get("contact") or ""
            if not out.get("license_no"):
                out["license_no"] = out.get("licence_no") or out.get("license") or ""
            return out

        # Bus details
        cur.execute(
            "SELECT bus_id, bus_no, capacity, bus_type, is_active, "
            "bus_register_no, pickup_id FROM bus_detail WHERE bus_id = %s LIMIT 1",
            (bus_id,)
        )
        bus = cur.fetchone()
        if not bus:
            # Many clients use BUS_ID as bus_no. Resolve gracefully.
            cur.execute(
                "SELECT bus_id, bus_no, capacity, bus_type, is_active, "
                "bus_register_no, pickup_id FROM bus_detail WHERE bus_no = %s LIMIT 1",
                (bus_id,)
            )
            bus = cur.fetchone()
        if bus:
            result["bus_info"] = _norm_row(bus)
        resolved_bus_id = str((bus or {}).get("bus_id") or bus_id).strip()

        # Driver details — adjust column names if your schema differs
        drv = None
        try:
            cur.execute(
                "SELECT * FROM driver_detail WHERE bus_id = %s LIMIT 1",
                (resolved_bus_id,)
            )
            drv = cur.fetchone()
        except Exception as e:
            logger.debug(f"driver_detail query: {e}")

        if not drv and bus:
            # Fallback 1: some datasets map drivers by pickup_id.
            pickup_id = bus.get("pickup_id")
            if pickup_id not in (None, "", 0, "0"):
                try:
                    cur.execute(
                        "SELECT * FROM driver_detail WHERE pickup_id = %s LIMIT 1",
                        (pickup_id,)
                    )
                    drv = cur.fetchone()
                except Exception as e:
                    logger.debug(f"driver_detail pickup query: {e}")

        if not drv and bus:
            # Fallback 2: some databases store driver bus reference as bus_no.
            bus_no = bus.get("bus_no")
            if bus_no not in (None, "", 0, "0"):
                try:
                    cur.execute(
                        "SELECT * FROM driver_detail WHERE bus_no = %s LIMIT 1",
                        (bus_no,)
                    )
                    drv = cur.fetchone()
                except Exception as e:
                    logger.debug(f"driver_detail bus_no query: {e}")

        if drv:
            result["driver_info"] = _norm_row(drv)

        cur.close(); c.close()
        logger.info(f"Bus info | bus_id={bus_id} "
                    f"bus_no={result['bus_info'].get('bus_no','?')} "
                    f"driver={result['driver_info'].get('driver_name', result['driver_info'].get('name','?'))}")
    except Exception as e:
        logger.error(f"fetch_bus_and_driver bus_id={bus_id}: {e}")
    return result


# =============================================================================
# BUS REGISTRY
# =============================================================================
def _get_or_create_bus(bus_id: str) -> dict:
    bus_id = str(bus_id).strip()
    with _bus_reg_lock:
        if bus_id not in _bus_registry:
            _bus_registry[bus_id] = {
                "bus_id":       bus_id,
                "bus_info":     {},
                "driver_info":  {},
                "stream_b64":   None,
                "stream_jpg":   None,
                "annotated_jpg": None,
                "annotated_ts":  None,
                "stream_ts":    None,
                "last_seen":    time.time(),
                "gps":          {},
                "session_stats":{"valid": 0, "unpaid": 0, "invalid": 0, "not_uni": 0},
                "registered_at": datetime.now().isoformat(),
            }
            threading.Thread(
                target=_load_bus_info_bg, args=(bus_id,), daemon=True
            ).start()
        else:
            _bus_registry[bus_id]["last_seen"] = time.time()
        return _bus_registry[bus_id]


def _load_bus_info_bg(bus_id: str):
    info = fetch_bus_and_driver(bus_id)
    with _bus_reg_lock:
        if bus_id in _bus_registry:
            _bus_registry[bus_id]["bus_info"]    = info["bus_info"]
            _bus_registry[bus_id]["driver_info"] = info["driver_info"]


def _update_bus_stats(bus_id: str, summary: dict):
    bus_id = str(bus_id).strip()
    with _bus_reg_lock:
        if bus_id in _bus_registry:
            s = _bus_registry[bus_id]["session_stats"]
            s["valid"]   += summary.get("valid_with_bus", 0)
            s["unpaid"]  += summary.get("unpaid", 0)
            s["invalid"] += summary.get("invalid", 0)
            s["not_uni"] += summary.get("not_uni", 0)


def _bus_is_active(bus_id: str) -> bool:
    with _bus_reg_lock:
        b = _bus_registry.get(str(bus_id))
        if b is None:
            return False
        return time.time() - b.get("last_seen", 0) < Config.BUS_INACTIVE_SECS


def _bus_snapshot(bus_id: str) -> dict:
    with _bus_reg_lock:
        b = _bus_registry.get(str(bus_id), {})
        snap = copy.deepcopy(b)
    snap["is_active"] = time.time() - snap.get("last_seen", 0) < Config.BUS_INACTIVE_SECS
    # Don't include stream frame in list (too large)
    snap.pop("stream_b64", None)
    return snap


# =============================================================================
# MODEL
# =============================================================================
def _init_model():
    logger.info("Loading InsightFace buffalo_l...")
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=Config.INSIGHT_CTX, det_size=Config.DET_SIZE)
    logger.info("Model loaded OK")
    return fa


# =============================================================================
# EMBEDDINGS
# =============================================================================
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _l2(v):
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)


def _bgr2b64(bgr, q=82):
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _emb_from_bgr(bgr):
    faces = face_app.get(bgr)
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return _l2(best.embedding.astype(np.float32))


PHOTOS_DIR       = Path("photos")
EMBEDDINGS_CACHE  = Path("embeddings_cache.npz")


def _scan_photo_folder() -> dict:
    """Scan the photos/ folder and return {gr_no: [list_of_paths]}."""
    by_gr = {}
    if not PHOTOS_DIR.exists():
        logger.warning(f"Photos folder not found: {PHOTOS_DIR.resolve()}")
        return by_gr
    for img_path in sorted(PHOTOS_DIR.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in _IMG_EXT:
            continue
        if img_path.name.startswith("."):
            continue
        gr = img_path.stem.strip()
        if gr:
            by_gr.setdefault(gr, []).append(img_path)
    return by_gr


def _load_cache() -> dict:
    """Load embeddings from the cache file. Returns {gr_no: embedding_array} or empty dict."""
    if not EMBEDDINGS_CACHE.exists():
        return {}
    try:
        data = np.load(str(EMBEDDINGS_CACHE), allow_pickle=False)
        cache = {}
        for key in data.files:
            cache[key] = data[key]
        logger.info(f"Loaded {len(cache)} embeddings from cache file")
        return cache
    except Exception as e:
        logger.warning(f"Failed to load embeddings cache: {e}")
        return {}


def _save_cache(store: dict):
    """Save embeddings dict to cache file."""
    try:
        np.savez(str(EMBEDDINGS_CACHE), **store)
        logger.info(f"Saved {len(store)} embeddings to cache file")
    except Exception as e:
        logger.error(f"Failed to save embeddings cache: {e}")


def precompute_embeddings():
    """
    Smart embedding computation with caching:
      1. Scan photos/ folder for all student images
      2. Load existing embeddings cache if available
      3. Compare: if counts match and all GR numbers exist -> skip (use cache)
      4. If new photos found -> compute only the new ones
      5. If photos removed -> remove their embeddings
      6. Save updated cache back to disk
    """
    logger.info("=" * 50)
    logger.info("  Embedding Manager — Smart Cache System")
    logger.info("=" * 50)

    # ── Step 1: Scan photos folder ────────────────────────────────────────
    by_gr = _scan_photo_folder()
    folder_count = len(by_gr)
    logger.info(f"Photos folder: {PHOTOS_DIR.resolve()}")
    logger.info(f"Found {folder_count} student photos in photos/ folder")

    if folder_count == 0:
        logger.warning("No photos found in photos/ folder! Embeddings will be empty.")
        with emb_lock:
            embedding_store.clear()
            enrollment_imgs.clear()
        return

    # ── Step 2: Load existing cache ───────────────────────────────────────
    cached = _load_cache()
    cache_count = len(cached)

    folder_grs = set(by_gr.keys())
    cache_grs  = set(cached.keys())

    new_grs     = folder_grs - cache_grs      # photos added
    removed_grs = cache_grs - folder_grs      # photos deleted
    existing_grs = folder_grs & cache_grs     # already cached

    logger.info(f"Cache has {cache_count} embeddings | Folder has {folder_count} photos")
    logger.info(f"  New: {len(new_grs)} | Removed: {len(removed_grs)} | Existing: {len(existing_grs)}")

    # ── Step 3: Check if cache is fully up-to-date ────────────────────────
    if len(new_grs) == 0 and len(removed_grs) == 0:
        logger.info("✅ Cache is up-to-date! No embedding computation needed.")
        # Load cached embeddings directly into memory
        store = {}
        imgs  = {}
        for gr_no in cached:
            store[gr_no] = cached[gr_no].reshape(1, -1)
            # Generate thumbnail from photo for display
            paths = by_gr.get(gr_no, [])
            if paths:
                bgr = cv2.imread(str(paths[0]))
                if bgr is not None:
                    imgs[gr_no] = _bgr2b64(bgr, Config.ENROLL_JPEG_Q)
        with emb_lock:
            embedding_store.clear(); embedding_store.update(store)
            enrollment_imgs.clear(); enrollment_imgs.update(imgs)
        logger.info(f"Loaded {len(store)} embeddings from cache (zero computation)")
        return

    # ── Step 4: Build updated store ───────────────────────────────────────
    store = {}
    imgs  = {}
    ok    = 0
    fail  = 0

    # 4a. Keep existing cached embeddings (no recomputation)
    for gr_no in existing_grs:
        store[gr_no] = cached[gr_no].reshape(1, -1)
        paths = by_gr.get(gr_no, [])
        if paths:
            bgr = cv2.imread(str(paths[0]))
            if bgr is not None:
                imgs[gr_no] = _bgr2b64(bgr, Config.ENROLL_JPEG_Q)
        ok += 1

    logger.info(f"Reused {len(existing_grs)} embeddings from cache")

    # 4b. Compute embeddings ONLY for new photos
    if new_grs:
        logger.info(f"Computing embeddings for {len(new_grs)} NEW photos...")
        for idx, gr_no in enumerate(sorted(new_grs), 1):
            paths = by_gr[gr_no]
            embs  = []
            thumb = None
            for p in paths:
                try:
                    bgr = cv2.imread(str(p))
                    if bgr is None:
                        fail += 1
                        continue
                    emb = _emb_from_bgr(bgr)
                    if emb is None:
                        fail += 1
                        continue
                    embs.append(emb)
                    if thumb is None:
                        thumb = bgr
                except Exception as e:
                    logger.debug(f"Error processing {p}: {e}")
                    fail += 1
                    continue

            if not embs:
                logger.debug(f"No embedding for student {gr_no}")
                fail += 1
                continue

            store[gr_no] = _l2(np.mean(np.stack(embs), axis=0)).reshape(1, -1)
            if thumb is not None:
                imgs[gr_no] = _bgr2b64(thumb, Config.ENROLL_JPEG_Q)
            ok += 1

            if idx % 10 == 0 or idx == len(new_grs):
                logger.info(f"  Progress: {idx}/{len(new_grs)} new embeddings computed")

    # 4c. Removed photos are simply not included in store (already excluded)
    if removed_grs:
        logger.info(f"Removed {len(removed_grs)} embeddings for deleted photos: {sorted(removed_grs)[:10]}...")

    # ── Step 5: Update memory store ───────────────────────────────────────
    with emb_lock:
        embedding_store.clear(); embedding_store.update(store)
        enrollment_imgs.clear(); enrollment_imgs.update(imgs)

    # ── Step 6: Save updated cache to disk ────────────────────────────────
    flat_cache = {gr: emb.reshape(-1) for gr, emb in store.items()}
    _save_cache(flat_cache)

    logger.info(f"Embeddings | total={len(store)} new={len(new_grs)} "
                f"reused={len(existing_grs)} removed={len(removed_grs)} failed={fail}")


# =============================================================================
# FACE DETECTION
# =============================================================================
def detect_all_faces(bgr):
    raw = face_app.get(bgr)
    if not raw:
        return []
    raw = sorted(raw, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    h, w = bgr.shape[:2]
    results = []
    for face in raw:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        pw = max(12, int((x2-x1)*0.25))
        ph = max(12, int((y2-y1)*0.25))
        cx1 = max(0, x1-pw); cy1 = max(0, y1-ph)
        cx2 = min(w, x2+pw); cy2 = min(h, y2+ph)
        crop = bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0 or min(crop.shape[:2]) < 10:
            continue
        quality = _face_quality(face, bgr, crop)
        results.append({
            "embedding": _l2(face.embedding.astype(np.float32)),
            "crop_b64":  _bgr2b64(crop, Config.ENROLL_JPEG_Q),
            "bbox":      [x1, y1, x2, y2],
            "kps":       face.kps.tolist() if getattr(face, "kps", None) is not None else None,
            "quality":   quality,
        })
    return results


# =============================================================================
# MATCH
# =============================================================================
def match_face(live_emb):
    with emb_lock:
        if not embedding_store:
            return None, None, 0.0, 0.0
        scores = {gr: float((mat @ live_emb).max()) for gr, mat in embedding_store.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_gr, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best - second
    if best >= Config.CONFIDENCE_THRESHOLD and margin >= Config.MARGIN_THRESHOLD:
        return best_gr, best_gr, best, second
    return best_gr, None, best, second


# =============================================================================
# BUS POLICY
# =============================================================================
def _has_bus(stu) -> bool:
    if Config.BUS_CHECK_MODE == "fee":
        return True
    pid = stu.get("pickup_id")
    return pid is not None and str(pid).strip() not in ("", "0", "null", "None")


def _fee_paid(stu) -> bool:
    return str(stu.get("fee_status", "")).strip().lower() == "paid"


# =============================================================================
# ASYNC DISK SAVE
# =============================================================================
def _save_worker():
    while True:
        try:
            bgr, folder, label = _save_q.get(timeout=1)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(
                os.path.join(folder, f"{label}_{ts}.jpg"), bgr,
                [cv2.IMWRITE_JPEG_QUALITY, Config.CAPTURE_JPEG_Q]
            )
            _save_q.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Save worker: {e}")


def _save(bgr, folder, label):
    try:
        _save_q.put_nowait((bgr.copy(), folder, label))
    except queue.Full:
        logger.warning("Save queue full")


# =============================================================================
# PROOF IMAGE
# =============================================================================
def _save_proof_to_disk(bgr, route: str, gr_no, stop_name) -> str:
    if bgr is None:
        return None
    try:
        today  = datetime.now().strftime("%Y-%m-%d")
        ts     = datetime.now().strftime("%H%M%S_%f")
        safe_gr   = str(gr_no or "unknown").replace("/", "-")[:20]
        safe_stop = str(stop_name or "unknown").replace(" ", "_").replace("/", "-")[:30]
        folder = Path(Config.PROOF_DIR) / today / route
        folder.mkdir(parents=True, exist_ok=True)
        fname  = f"{safe_gr}_{safe_stop}_{ts}.jpg"
        fpath  = folder / fname
        cv2.imwrite(str(fpath), bgr, [cv2.IMWRITE_JPEG_QUALITY, Config.CAPTURE_JPEG_Q])
        return str(fpath)
    except Exception as e:
        logger.error(f"Proof save failed: {e}")
        return None


def _cleanup_old_proofs():
    import shutil
    def _run():
        while True:
            try:
                cutoff = datetime.now() - timedelta(days=Config.PROOF_RETAIN_DAYS)
                base   = Path(Config.PROOF_DIR)
                if base.exists():
                    for day_dir in sorted(base.iterdir()):
                        if not day_dir.is_dir():
                            continue
                        try:
                            if datetime.strptime(day_dir.name, "%Y-%m-%d") < cutoff:
                                shutil.rmtree(str(day_dir))
                                logger.info(f"Proof cleanup | deleted: {day_dir.name}")
                        except ValueError:
                            pass
            except Exception as e:
                logger.error(f"Proof cleanup error: {e}")
            time.sleep(3600)
    threading.Thread(target=_run, daemon=True, name="proof-cleanup").start()


# =============================================================================
# ENROLLMENT IMAGE
# =============================================================================
def _enroll_b64(gr):
    with emb_lock:
        b64 = enrollment_imgs.get(str(gr), "")
    if not b64 and gr:
        try:
            # Look in local photos/ folder
            candidates = []
            sub = PHOTOS_DIR / str(gr)
            if sub.is_dir():
                candidates.extend(sorted(sub.glob("*")))
            for ext in (".jpg", ".jpeg", ".png"):
                candidates.append(PHOTOS_DIR / (str(gr) + ext))
            for p in candidates:
                if p.is_file() and p.suffix.lower() in _IMG_EXT:
                    data = p.read_bytes()
                    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
                    b64 = f"data:{mime};base64," + base64.b64encode(data).decode()
                    break
        except:
            pass
    return b64


# =============================================================================
# DAILY SLOT COOLDOWN
# =============================================================================
def _on_cd(gr: str) -> bool:
    with cd_lock:
        rec = student_cooldown.get(gr)
        if rec is None:
            return False
        if time.time() - rec.get("last_ts", 0) < 300:
            return True
        slot = _get_slot()
        today = _today()
        return rec.get(slot) == today


def _set_cd(gr: str):
    slot = _get_slot(); today = _today()
    with cd_lock:
        rec = student_cooldown.get(gr, {})
        rec["last_ts"] = time.time()
        rec[slot] = today
        student_cooldown[gr] = rec


def _cd_left(gr: str) -> int:
    with cd_lock:
        rec = student_cooldown.get(gr)
        if rec is None:
            return 0
        hard = max(0, int(300 - (time.time() - rec.get("last_ts", 0))))
        if hard > 0:
            return hard
        slot = _get_slot(); today = _today()
        if rec.get(slot) != today:
            return 0
        now = datetime.now()
        if slot == "morning":
            next_open = now.replace(hour=14, minute=0, second=0, microsecond=0)
        else:
            next_open = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(0, int((next_open - now).total_seconds()))


# =============================================================================
# THROTTLE
# =============================================================================
def _throttle_ok(key: str) -> bool:
    now = time.time()
    with _throttle_lock:
        if now - _throttle.get(key, 0.0) < Config.THROTTLE_SECS:
            return False
        _throttle[key] = now
    return True


# =============================================================================
# STORE FUNCTIONS (now include bus_id)
# =============================================================================
def _store_not_uni(crop_b64, conf, candidate_gr, msg, location=None,
                   frame_bgr=None, bus_id=None):
    rec = {
        "route": "not_uni_student", "event": "unknown_person",
        "timestamp": datetime.now().isoformat(),
        "confidence": round(float(conf), 4) if conf else None,
        "candidate_gr": candidate_gr, "message": msg,
        "captured_b64": crop_b64, "enrollment_b64": None,
        "location": location or {}, "bus_id": str(bus_id) if bus_id else None,
        "proof_file": _save_proof_to_disk(frame_bgr, "not_uni_student",
                          candidate_gr, location.get("stop_name") if location else None),
    }
    with nu_lock:
        not_uni_store.insert(0, rec)
        if len(not_uni_store) > Config.NOT_UNI_STORE_MAX:
            not_uni_store.pop()
    _sse_broadcast("not_uni", "1")


def _store_invalid(reason, crop_b64, conf, gr, msg, stu=None, location=None,
                   frame_bgr=None, bus_id=None):
    enr = _enroll_b64(str(gr) if gr else "")
    rec = {
        "route": "invalid_alerts", "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "confidence": round(float(conf), 4) if conf else None,
        "gr_no": str(gr) if gr else None,
        "enrollment_no": str(stu["enrollment_no"]) if stu else None,
        "student_name": stu["student_name"] if stu else None,
        "department": stu["department"] if stu else None,
        "semester": str(stu["semester"]) if stu else None,
        "shift": stu["shift"] if stu else None,
        "fee_status": stu["fee_status"] if stu else None,
        "pickup_id": str(stu.get("pickup_id", "")) if stu else None,
        "message": msg, "captured_b64": crop_b64,
        "enrollment_b64": enr, "location": location or {},
        "bus_id": str(bus_id) if bus_id else None,
        "proof_file": _save_proof_to_disk(frame_bgr, "invalid_alerts",
                          gr, location.get("stop_name") if location else None),
    }
    with inv_lock:
        invalid_store.insert(0, rec)
        if len(invalid_store) > Config.INVALID_STORE_MAX:
            invalid_store.pop()
    _sse_broadcast("invalid", reason)


def _store_unpaid(stu, crop_b64, location=None, frame_bgr=None, bus_id=None):
    gr = str(stu.get("gr_no", ""))
    rec = {
        "route": "unpaid_students", "event": "unpaid_fee_detected",
        "timestamp": datetime.now().isoformat(),
        "gr_no": gr,
        "enrollment_no": str(stu.get("enrollment_no", "")),
        "student_name": stu.get("student_name", ""),
        "department": stu.get("department", ""),
        "semester": str(stu.get("semester", "")),
        "shift": stu.get("shift", ""),
        "fee_status": stu.get("fee_status", ""),
        "pickup_id": str(stu.get("pickup_id", "")),
        "captured_b64": crop_b64, "enrollment_b64": _enroll_b64(gr),
        "location": location or {}, "bus_id": str(bus_id) if bus_id else None,
        "proof_file": _save_proof_to_disk(frame_bgr, "unpaid_students",
                          gr, location.get("stop_name") if location else None),
    }
    with unpaid_lock:
        unpaid_store[gr] = rec
        if len(unpaid_store) > Config.UNPAID_STORE_MAX:
            del unpaid_store[next(iter(unpaid_store))]
    _sse_broadcast("unpaid", gr)


def _store_valid(stu, crop_b64, location=None, frame_bgr=None, bus_id=None):
    gr = str(stu.get("gr_no", ""))
    rec = {
        "route": "valid_with_bus", "event": "access_granted",
        "timestamp": datetime.now().isoformat(),
        "gr_no": gr,
        "enrollment_no": str(stu.get("enrollment_no", "")),
        "student_name": stu.get("student_name", ""),
        "department": stu.get("department", ""),
        "semester": str(stu.get("semester", "")),
        "shift": stu.get("shift", ""),
        "fee_status": stu.get("fee_status", ""),
        "pickup_id": str(stu.get("pickup_id", "")),
        "captured_b64": crop_b64, "enrollment_b64": _enroll_b64(gr),
        "location": location or {}, "bus_id": str(bus_id) if bus_id else None,
        "proof_file": _save_proof_to_disk(frame_bgr, "valid_with_bus",
                          gr, location.get("stop_name") if location else None),
    }
    with valid_lock:
        valid_store.insert(0, rec)
        if len(valid_store) > Config.VALID_STORE_MAX:
            valid_store.pop()
    logger.info(f"[E] GRANTED | gr={gr} name={stu.get('student_name')} "
                f"bus={bus_id} stop={location and location.get('stop_name')}")
    _sse_broadcast("valid", gr)


# =============================================================================
# DASHBOARD PUSH
# =============================================================================
def _push_scan(new_scan):
    global latest_scan, _last_scan_ts
    new_p = max((_PRIO.get(r.get("status", ""), 0) for r in new_scan.get("results", [])), default=0)
    with scan_lock:
        cur_p = max((_PRIO.get(r.get("status", ""), 0) for r in latest_scan.get("results", [])), default=0)
        if time.time() - _last_scan_ts < Config.RESULT_HOLD_SECS and new_p < cur_p:
            return
        latest_scan = copy.deepcopy(new_scan)
        latest_scan["is_idle"] = False
        _last_scan_ts = time.time()


# =============================================================================
# DECISION TREE
# =============================================================================
def process_one_face(emb, crop_b64, frame, location, skip_grs=None, bus_id=None, quality=None) -> dict:
    ts = datetime.now().isoformat()

    if quality and not quality.get("ok", True):
        msg = f"Low quality face ({quality.get('reason') or 'blur / small face'})"
        return {
            "status": "low_quality", "route": "blurred_face", "category": "low_quality",
            "gr_no": None, "enrollment_no": None, "name": "Blurry / Low Quality Face",
            "display_name": "Blurry / Low Quality Face",
            "state_label": _status_label("low_quality", "low_quality"),
            "label": _make_label("Blurry / Low Quality Face", "low_quality", "low_quality"),
            "department": None, "semester": None, "shift": None,
            "fee_status": None, "pickup_id": None,
            "confidence": 0.0, "margin": 0.0,
            "message": msg,
            "timestamp": ts, "captured_b64": crop_b64, "enrollment_b64": None,
            "location": location, "bus_id": str(bus_id) if bus_id else None,
            "quality": quality,
        }

    candidate_gr, matched_gr, best, second = match_face(emb)
    margin = best - second

    # Route A: No match
    if matched_gr is None:
        key = f"not_uni_{candidate_gr or 'unk'}"
        result = {
            "status": "not_uni", "route": "not_uni_student", "category": "not_uni_student",
            "gr_no": None, "enrollment_no": None, "name": "Unknown Person",
            "display_name": "Unknown Person",
            "state_label": _status_label("not_uni", "not_uni_student"),
            "label": _make_label("Unknown Person", "not_uni", "not_uni_student"),
            "department": None, "semester": None, "shift": None,
            "fee_status": None, "pickup_id": None,
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": f"No match (score={best:.3f}, margin={margin:.3f})",
            "timestamp": ts, "captured_b64": crop_b64, "enrollment_b64": None,
            "location": location, "bus_id": str(bus_id) if bus_id else None,
        }
        if _throttle_ok(key):
            _save(frame, Config.DIR_NOT_UNI, "not_uni")
            _store_not_uni(crop_b64, best, candidate_gr, result["message"],
                           location, frame_bgr=frame, bus_id=bus_id)
        return result

    # Cooldown
    if _on_cd(matched_gr):
        return {
            "status": "on_cooldown", "route": "cooldown", "category": "cooldown",
            "gr_no": matched_gr, "enrollment_no": None, "name": None,
            "display_name": matched_gr,
            "state_label": _status_label("on_cooldown", "cooldown"),
            "label": _make_label(matched_gr, "on_cooldown", "cooldown"),
            "department": None, "semester": None, "shift": None,
            "fee_status": None, "pickup_id": None, "on_cooldown": True,
            "cooldown_secs": _cd_left(matched_gr),
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": f"GR {matched_gr} on cooldown",
            "timestamp": ts, "captured_b64": crop_b64,
            "enrollment_b64": _enroll_b64(matched_gr), "location": location,
            "bus_id": str(bus_id) if bus_id else None,
        }

    if skip_grs and matched_gr in skip_grs:
        return {
            "status": "on_cooldown", "route": "cooldown", "category": "shift_skip",
            "gr_no": matched_gr, "on_cooldown": True,
            "display_name": matched_gr,
            "state_label": _status_label("on_cooldown", "cooldown"),
            "label": _make_label(matched_gr, "on_cooldown", "cooldown"),
            "cooldown_secs": _cd_left(matched_gr),
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": f"GR {matched_gr} already validated this shift",
            "timestamp": ts, "captured_b64": crop_b64,
            "enrollment_b64": _enroll_b64(matched_gr), "location": location,
            "bus_id": str(bus_id) if bus_id else None,
        }

    enr_b64 = _enroll_b64(matched_gr)
    stu = fetch_student(matched_gr)

    # Route B: Photo matched, not in DB
    if stu is None:
        msg = f"GR '{matched_gr}' photo matched but NOT in students_detail"
        result = {
            "status": "invalid_database", "route": "invalid_alerts", "category": "not_in_db",
            "gr_no": matched_gr, "enrollment_no": None, "name": None,
            "display_name": f"GR {matched_gr}",
            "state_label": _status_label("invalid_database", "not_in_db"),
            "label": _make_label(f"GR {matched_gr}", "invalid_database", "not_in_db"),
            "department": None, "semester": None, "shift": None,
            "fee_status": None, "pickup_id": None,
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": msg, "timestamp": ts, "captured_b64": crop_b64,
            "enrollment_b64": enr_b64, "location": location,
            "bus_id": str(bus_id) if bus_id else None,
        }
        with _nodb_lock:
            first = matched_gr not in _nodb_reported
            _nodb_reported[matched_gr] = time.time()
        if first:
            _save(frame, Config.DIR_INVALID, f"nodb_{matched_gr}")
            _store_invalid("not_in_db", crop_b64, best, matched_gr, msg,
                           location=location, frame_bgr=frame, bus_id=bus_id)
        _set_cd(matched_gr)
        return result

    # Route C: In DB, no bus
    if not _has_bus(stu):
        msg = (f"{stu['student_name']} (GR:{matched_gr}) has no bus subscription")
        result = {
            "status": "invalid_person", "route": "invalid_alerts", "category": "no_bus_policy",
            "gr_no": str(stu["gr_no"]), "enrollment_no": str(stu["enrollment_no"]),
            "name": stu["student_name"], "department": stu["department"],
            "display_name": stu["student_name"],
            "state_label": _status_label("invalid_person", "no_bus_policy"),
            "label": _make_label(stu["student_name"], "invalid_person", "no_bus_policy"),
            "semester": str(stu["semester"]), "shift": stu["shift"],
            "fee_status": stu["fee_status"], "pickup_id": str(stu.get("pickup_id", "")),
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": msg, "timestamp": ts, "captured_b64": crop_b64,
            "enrollment_b64": enr_b64, "location": location,
            "bus_id": str(bus_id) if bus_id else None,
        }
        _save(frame, Config.DIR_INVALID, f"nobus_{matched_gr}")
        _store_invalid("no_bus_policy", crop_b64, best, matched_gr, msg,
                       stu=stu, location=location, frame_bgr=frame, bus_id=bus_id)
        _set_cd(matched_gr)
        return result

    # Route D: Bus, fee UNPAID
    if not _fee_paid(stu):
        msg = f"Bus user — FEE UNPAID: {stu['student_name']} (GR:{matched_gr})"
        result = {
            "status": "valid_without_bus", "route": "unpaid_students", "category": "unpaid_fee",
            "gr_no": str(stu["gr_no"]), "enrollment_no": str(stu["enrollment_no"]),
            "name": stu["student_name"], "department": stu["department"],
            "display_name": stu["student_name"],
            "state_label": _status_label("valid_without_bus", "unpaid_fee"),
            "label": _make_label(stu["student_name"], "valid_without_bus", "unpaid_fee"),
            "semester": str(stu["semester"]), "shift": stu["shift"],
            "fee_status": stu["fee_status"], "pickup_id": str(stu.get("pickup_id", "")),
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": msg, "timestamp": ts, "captured_b64": crop_b64,
            "enrollment_b64": enr_b64, "location": location,
            "bus_id": str(bus_id) if bus_id else None,
        }
        _save(frame, Config.DIR_WITHOUT_BUS, matched_gr)
        _store_unpaid(stu, crop_b64, location=location, frame_bgr=frame, bus_id=bus_id)
        _set_cd(matched_gr)
        return result

    # Route E: ACCESS GRANTED
    msg = f"ACCESS GRANTED — {stu['student_name']} (GR:{matched_gr}, fee PAID)"
    result = {
        "status": "valid_with_bus", "route": "valid_with_bus", "category": "valid_with_bus",
        "gr_no": str(stu["gr_no"]), "enrollment_no": str(stu["enrollment_no"]),
        "name": stu["student_name"], "department": stu["department"],
        "display_name": stu["student_name"],
        "state_label": _status_label("valid_with_bus", "valid_with_bus"),
        "label": _make_label(stu["student_name"], "valid_with_bus", "valid_with_bus"),
        "semester": str(stu["semester"]), "shift": stu["shift"],
        "fee_status": stu["fee_status"], "pickup_id": str(stu.get("pickup_id", "")),
        "confidence": round(float(best), 4), "margin": round(float(margin), 4),
        "message": msg, "timestamp": ts, "captured_b64": crop_b64,
        "enrollment_b64": enr_b64, "location": location,
        "bus_id": str(bus_id) if bus_id else None,
    }
    _save(frame, Config.DIR_WITH_BUS, matched_gr)
    _store_valid(stu, crop_b64, location=location, frame_bgr=frame, bus_id=bus_id)
    _set_cd(matched_gr)
    return result


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/")
def dashboard():
    return send_from_directory("templates", "index.html")


@app.route("/live_frame")
def live_frame_route():
    with _live_lock:
        b64 = _live_frame_b64
    return jsonify({"frame": b64, "ready": b64 is not None})


# ── Multi-bus: Register bus ────────────────────────────────────────────────
@app.route("/register_bus", methods=["POST", "OPTIONS"])
def register_bus():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    d = request.get_json(silent=True) or {}
    bus_id = str(d.get("bus_id", "")).strip()
    if not bus_id:
        return jsonify({"error": "bus_id required"}), 400
    bus = _get_or_create_bus(bus_id)
    logger.info(f"Bus registered | bus_id={bus_id}")
    resp = {
        "status": "registered",
        "bus_id": bus_id,
        **bus.get("bus_info", {}),
        **{f"driver_{k}": v for k, v in bus.get("driver_info", {}).items()},
    }
    return jsonify(resp)


# ── Multi-bus: Push stream frame from Pi ──────────────────────────────────
@app.route("/stream_frame", methods=["POST", "OPTIONS"])
def push_stream_frame():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    bus_id = (request.form.get("bus_id") or "0").strip()
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    raw = request.files["image"].read()
    frame_b64 = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    ts = datetime.now().isoformat()

    bus = _get_or_create_bus(bus_id)
    with _bus_reg_lock:
        bus["stream_b64"] = frame_b64
        bus["stream_jpg"] = raw
        bus["stream_ts"]  = ts
        bus["last_seen"]  = time.time()

    # Update global live frame too
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is not None:
        with _live_lock:
            global _live_frame, _live_frame_b64
            _live_frame     = frame
            _live_frame_b64 = frame_b64

    return jsonify({"ok": True, "bus_id": bus_id, "timestamp": ts})


# ── Multi-bus: Get stream frame per bus ───────────────────────────────────
@app.route("/live_frame_bus", methods=["GET", "OPTIONS"])
def live_frame_bus():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    bus_id = request.args.get("bus_id", "").strip()
    with _bus_reg_lock:
        bus = _bus_registry.get(bus_id, {})
        frame_b64 = bus.get("stream_b64")
        ts        = bus.get("stream_ts")
    age_ok = True
    if ts:
        try:
            age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
            age_ok = age < Config.STREAM_FRAME_TTL
        except:
            pass
    return jsonify({
        "bus_id":    bus_id,
        "frame":     frame_b64 if age_ok else None,
        "timestamp": ts,
        "ready":     frame_b64 is not None and age_ok,
    })


@app.route("/live_frame_bus.jpg", methods=["GET", "OPTIONS"])
def live_frame_bus_jpg():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    bus_id = request.args.get("bus_id", "").strip()
    with _bus_reg_lock:
        bus = _bus_registry.get(bus_id, {})
        frame = bus.get("annotated_jpg") or bus.get("stream_jpg")
        ts = bus.get("annotated_ts") or bus.get("stream_ts")
    if not frame:
        return ("", 204)
    age_ok = True
    if ts:
        try:
            age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
            age_ok = age < Config.STREAM_FRAME_TTL
        except:
            pass
    if not age_ok:
        return ("", 204)
    return Response(frame, mimetype="image/jpeg", headers={
        "Cache-Control": "no-store,no-cache,must-revalidate",
        "Pragma": "no-cache",
        "X-Frame-Timestamp": ts or "",
        "X-Bus-Id": bus_id,
    })


# ── Multi-bus: List all active buses ──────────────────────────────────────
@app.route("/buses", methods=["GET", "OPTIONS"])
def list_buses():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    now = time.time()
    with _bus_reg_lock:
        buses = []
        for bus_id, b in _bus_registry.items():
            age = now - b.get("last_seen", 0)
            snap = {
                "bus_id":          bus_id,
                "bus_info":        b.get("bus_info", {}),
                "driver_info":     b.get("driver_info", {}),
                "stream_ts":       b.get("stream_ts"),
                "last_seen":       b.get("last_seen"),
                "is_active":       age < Config.BUS_INACTIVE_SECS,
                "age_secs":        int(age),
                "session_stats":   b.get("session_stats", {}),
                "gps":             b.get("gps", {}),
                "registered_at":   b.get("registered_at"),
            }
            buses.append(snap)
    buses.sort(key=lambda x: (not x["is_active"], x["bus_id"]))
    return jsonify({
        "total":  len(buses),
        "active": sum(1 for b in buses if b["is_active"]),
        "buses":  buses,
    })


@app.route("/pickup_points", methods=["GET", "OPTIONS"])
def get_pickup_points():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        c = _get_db()
        cur = c.cursor(dictionary=True)
        cur.execute("""
            SELECT pickup_id, pickup_name, latitude, longitude, city, state, country
            FROM pickup_points
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
              AND latitude != 0 AND longitude != 0
            ORDER BY pickup_id ASC
        """)
        rows = cur.fetchall()
        cur.close(); c.close()
        stops = []
        for row in rows:
            try:
                stops.append({
                    "pickup_id":    int(row["pickup_id"]),
                    "name":         str(row["pickup_name"]).strip(),
                    "lat":          float(row["latitude"]),
                    "lon":          float(row["longitude"]),
                    "city":         str(row.get("city", "") or "").strip(),
                    "state":        str(row.get("state", "") or "").strip(),
                    "country":      str(row.get("country", "") or "").strip(),
                    "display_name": "",
                })
            except:
                continue
        return jsonify({"total": len(stops), "stops": stops})
    except Exception as e:
        logger.error(f"/pickup_points: {e}")
        return jsonify({"error": str(e), "total": 0, "stops": []}), 500


@app.route("/bus_location", methods=["GET", "OPTIONS"])
def bus_location():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with _bus_loc_lock:
        return jsonify(copy.deepcopy(_bus_location))


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    t0 = time.time()
    if not _upload_sem.acquire(blocking=True, timeout=5):
        return jsonify({"status": "busy", "message": "Server busy"}), 503
    try:
        bus_id   = (request.form.get("bus_id") or "0").strip()
        location = _parse_location(request.form)
        _update_bus_location(location)

        # Update bus GPS location
        bus = _get_or_create_bus(bus_id)
        with _bus_reg_lock:
            bus["gps"] = {
                "lat": location.get("gps_lat"),
                "lon": location.get("gps_lon"),
                "stop_name": location.get("stop_name"),
                "updated_at": datetime.now().isoformat(),
            }
            bus["last_seen"] = time.time()

        skip_raw = request.form.get("skip_gr_list", "")
        skip_grs = set(g.strip() for g in skip_raw.split(",") if g.strip())

        logger.info(f"Upload | bus={bus_id} stop='{location.get('stop_name', '?')}' "
                    f"gps=({location.get('gps_lat')},{location.get('gps_lon')})")

        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No image"}), 400
        raw   = request.files["image"].read()
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "error", "message": "Cannot decode image"}), 400

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, Config.LIVE_JPEG_Q])
        frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        with _live_lock:
            global _live_frame, _live_frame_b64
            _live_frame     = frame
            _live_frame_b64 = frame_b64

        detected = detect_all_faces(frame)
        logger.info(f"Upload | bus={bus_id} {len(detected)} face(s) | {_ms(t0)}")

        if not detected:
            return jsonify({
                "status": "no_face", "face_count": 0, "results": [], "is_idle": False,
                "summary": {"valid_with_bus": 0, "unpaid": 0, "invalid": 0, "not_uni": 0, "cooldown": 0},
                "location": location, "bus_id": bus_id,
                "message": "No face detected", "timestamp": datetime.now().isoformat(),
                "captured_b64": frame_b64,
            })

        results = []
        for idx, fi in enumerate(detected):
            r = process_one_face(fi["embedding"], fi["crop_b64"], frame,
                                 location, skip_grs=skip_grs, bus_id=bus_id,
                                 quality=fi.get("quality"))
            r["face_index"] = idx
            r["bbox"]       = fi["bbox"]
            if fi.get("quality"):
                r["quality"] = fi["quality"]
                if fi.get("kps") is not None:
                    r["kps"] = fi["kps"]
            results.append(r)

        n = len(results)
        summary = {
            "valid_with_bus": sum(1 for r in results if r["status"] == "valid_with_bus"),
            "unpaid":         sum(1 for r in results if r["status"] == "valid_without_bus"),
            "invalid":        sum(1 for r in results if r["status"] in ("invalid_person", "invalid_database")),
            "not_uni":        sum(1 for r in results if r["status"] == "not_uni"),
            "low_quality":    sum(1 for r in results if r["status"] == "low_quality"),
            "cooldown":       sum(1 for r in results if r["status"] == "on_cooldown"),
        }

        # Update bus session stats
        _update_bus_stats(bus_id, summary)

        annotated_frame = _annotate_scan_frame(frame, results)
        if annotated_frame is not None:
            _, ann_buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, Config.CAPTURE_JPEG_Q])
            ann_b64 = "data:image/jpeg;base64," + base64.b64encode(ann_buf.tobytes()).decode()
            with _bus_reg_lock:
                bus = _get_or_create_bus(bus_id)
                bus["annotated_jpg"] = ann_buf.tobytes()
                bus["annotated_ts"] = datetime.now().isoformat()
                bus["stream_b64"] = ann_b64

        scan = {
            "is_idle": False, "multi_face": n > 1, "face_count": n,
            "results": results, "summary": summary, "location": location,
            "timestamp": datetime.now().isoformat(), "captured_b64": frame_b64,
            "bus_id": bus_id,
        }
        _push_scan(scan)

        import json as _json
        _sse_broadcast("scan", _json.dumps({
            "face_count": n, "summary": summary,
            "timestamp": scan["timestamp"],
            "stop_name": location.get("stop_name") or "",
            "bus_id": bus_id,
        }))

        logger.info(f"Done | bus={bus_id} faces={n} granted={summary['valid_with_bus']} "
                    f"unpaid={summary['unpaid']} invalid={summary['invalid']} | {_ms(t0)}")
        return jsonify(scan)
    finally:
        _upload_sem.release()


@app.route("/upload_status", methods=["GET", "OPTIONS"])
def upload_status():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with scan_lock:
        return jsonify(copy.deepcopy(latest_scan))


@app.route("/valid_students", methods=["GET", "OPTIONS"])
def get_valid_students():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:    limit = int(request.args.get("limit", 200))
    except: limit = 200
    bus_filter = request.args.get("bus_id", "").strip()
    with valid_lock:
        data = [r for r in valid_store if not bus_filter or r.get("bus_id") == bus_filter]
        data = data[:limit]
        total = len(valid_store)
        if request.args.get("clear", "0") == "1":
            valid_store.clear()
    return jsonify({"total": total, "route": "valid_with_bus", "students": data})


@app.route("/valid_students/clear", methods=["POST", "OPTIONS"])
def clear_valid():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with valid_lock:
        n = len(valid_store); valid_store.clear()
    return jsonify({"status": "cleared", "removed": n})


@app.route("/unpaid_students", methods=["GET", "OPTIONS"])
def get_unpaid_students():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    gr = request.args.get("gr_no", "").strip()
    bus_filter = request.args.get("bus_id", "").strip()
    with unpaid_lock:
        if gr:
            data = [unpaid_store[gr]] if gr in unpaid_store else []
        elif bus_filter:
            data = [v for v in unpaid_store.values() if v.get("bus_id") == bus_filter]
        else:
            data = list(unpaid_store.values())
        if request.args.get("clear", "0") == "1":
            if gr: unpaid_store.pop(gr, None)
            else:  unpaid_store.clear()
    data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify({"total": len(data), "route": "unpaid_students", "unpaid_students": data})


@app.route("/unpaid_students/clear", methods=["POST", "OPTIONS"])
def clear_unpaid():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with unpaid_lock:
        n = len(unpaid_store); unpaid_store.clear()
    return jsonify({"status": "cleared", "removed": n})


@app.route("/invalid_alerts", methods=["GET", "OPTIONS"])
def get_invalid_alerts():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    reason     = request.args.get("reason", "").strip()
    bus_filter = request.args.get("bus_id", "").strip()
    try:    limit = int(request.args.get("limit", Config.INVALID_STORE_MAX))
    except: limit = Config.INVALID_STORE_MAX
    with inv_lock:
        data = list(invalid_store)
        if reason:     data = [r for r in data if r.get("reason") == reason]
        if bus_filter: data = [r for r in data if r.get("bus_id") == bus_filter]
        data = data[:limit]
        if request.args.get("clear", "0") == "1":
            invalid_store.clear()
        counts = {"not_in_db": 0, "no_bus_policy": 0}
        for r in invalid_store:
            k = r.get("reason", "")
            if k in counts: counts[k] += 1
    return jsonify({"total": len(invalid_store), "counts_by_type": counts,
                    "showing": len(data), "route": "invalid_alerts", "alerts": data})


@app.route("/invalid_alerts/clear", methods=["POST", "OPTIONS"])
def clear_invalid():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with inv_lock:
        n = len(invalid_store); invalid_store.clear()
    return jsonify({"status": "cleared", "removed": n})


@app.route("/not_uni_student", methods=["GET", "OPTIONS"])
def get_not_uni():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    bus_filter = request.args.get("bus_id", "").strip()
    try:    limit = int(request.args.get("limit", Config.NOT_UNI_STORE_MAX))
    except: limit = Config.NOT_UNI_STORE_MAX
    with nu_lock:
        data = list(not_uni_store)
        if bus_filter: data = [r for r in data if r.get("bus_id") == bus_filter]
        data = data[:limit]
        total = len(not_uni_store)
        if request.args.get("clear", "0") == "1":
            not_uni_store.clear()
    return jsonify({"total": total, "route": "not_uni_student", "not_uni_students": data})


@app.route("/not_uni_student/clear", methods=["POST", "OPTIONS"])
def clear_not_uni():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with nu_lock:
        n = len(not_uni_store); not_uni_store.clear()
    return jsonify({"status": "cleared", "removed": n})


@app.route("/cooldown_status", methods=["GET", "OPTIONS"])
def cooldown_status():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    today = _today()
    with cd_lock:
        active = {}
        for gr, rec in student_cooldown.items():
            m = rec.get("morning") == today
            e = rec.get("evening") == today
            if m or e:
                active[gr] = {
                    "morning_done": m, "evening_done": e,
                    "next_slot_secs": _cd_left(gr),
                    "last_seen": datetime.fromtimestamp(rec.get("last_ts", 0)).isoformat(),
                }
    return jsonify({"today": today, "current_slot": _get_slot(),
                    "max_per_day": 2, "count": len(active), "active": active})


@app.route("/validated_today", methods=["GET", "OPTIONS"])
def validated_today():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    today = _today(); slot = _get_slot()
    with cd_lock:
        validated = [gr for gr, rec in student_cooldown.items()
                     if rec.get(slot) == today]
    return jsonify({"today": today, "slot": slot,
                    "count": len(validated), "gr_list": validated})


@app.route("/reload_embeddings", methods=["POST", "OPTIONS"])
def reload_embeddings():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if request.headers.get("X-Reload-Secret", "") != os.environ.get("RELOAD_SECRET", "changeme"):
        return jsonify({"error": "Forbidden"}), 403
    threading.Thread(target=precompute_embeddings, daemon=True).start()
    return jsonify({"status": "reloading"})


@app.route("/cache/clear_student", methods=["POST", "OPTIONS"])
def clear_cache():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    d = request.get_json(silent=True) or {}
    gr = d.get("gr_no", "").strip()
    if gr:
        invalidate_student(gr)
        return jsonify({"status": "cleared", "gr_no": gr})
    with _stu_cache_lock:
        n = len(_stu_cache); _stu_cache.clear()
    return jsonify({"status": "all_cleared", "removed": n})


@app.route("/events")
def sse_stream():
    def _gen():
        q = queue.Queue(maxsize=50)
        with _sse_clients_lock:
            _sse_clients.append(q)
        try:
            yield "event: connected\ndata: ok\n\n"
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield "event: heartbeat\ndata: ping\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_clients_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)
    return Response(_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with emb_lock:
        n = len(embedding_store)
    with _bus_reg_lock:
        bus_count = len(_bus_registry)
        active_buses = sum(
            1 for b in _bus_registry.values()
            if time.time() - b.get("last_seen", 0) < Config.BUS_INACTIVE_SECS
        )
    return jsonify({
        "status": "ok", "version": "12.0.0",
        "model_loaded": face_app is not None,
        "students_loaded": n,
        "bus_check_mode": Config.BUS_CHECK_MODE,
        "buses_registered": bus_count,
        "buses_active": active_buses,
        "timestamp": datetime.now().isoformat(),
    })


# =============================================================================
# STARTUP
# =============================================================================
def startup():
    global face_app
    logger.info("=" * 64)
    logger.info("  TransBuddy Server v12.0.0 — Marwadi University")
    logger.info("  Multi-Bus Face Verification System")
    logger.info("=" * 64)
    _init_db_pool()
    test_db()
    face_app = _init_model()
    precompute_embeddings()
    threading.Thread(target=_save_worker, daemon=True, name="save").start()
    _cleanup_old_proofs()
    logger.info("  Listening on http://0.0.0.0:5000")
    logger.info("=" * 64)


startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)