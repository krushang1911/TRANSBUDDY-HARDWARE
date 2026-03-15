#!/usr/bin/env python3
"""
TransBuddy Bus Face Verification Server — v11.0.0
Marwadi University

DECISION TREE (per detected face — strict single route):
  A. Face not matched to any photo  -> /not_uni_student     ONLY
  B. Photo matched, GR not in DB    -> /invalid_alerts      ONLY (not_in_db)
  C. In DB, no bus subscription     -> /invalid_alerts      ONLY (no_bus_policy)
  D. Has bus, fee UNPAID            -> /unpaid_students     ONLY
  E. Has bus, fee PAID              -> /valid_students      ONLY (ACCESS GRANTED)

DAILY VALIDATION LIMIT:
  Morning slot = before 14:00  -> max 1 validation per student
  Evening slot = 14:00 onwards -> max 1 validation per student
  Total = max 2 validations per student per day
  5-minute hard block against rapid re-triggers

BUS_CHECK_MODE:
  "fee"       = ALL DB students are bus users; fee_status decides route D or E
  "pickup_id" = only students with pickup_id set have bus access
"""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import copy
import os
import base64
import time
import logging
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from flask import Flask, request, jsonify, send_from_directory, Response
from insightface.app import FaceAnalysis


# ── LOGGING ───────────────────────────────────────────────────
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

    DIR_PHOTOS      = "photos"
    DIR_WITH_BUS    = "captures/with_bus"
    DIR_WITHOUT_BUS = "captures/without_bus"
    DIR_INVALID     = "captures/invalid_captures"
    DIR_NOT_UNI     = "captures/not_uni_student"

    DB_HOST      = "127.0.0.1"
    DB_PORT      = 3306
    DB_USER      = "root"
    DB_PASSWORD  = ""
    DB_NAME      = "transbuddy_db_1"
    DB_POOL_SIZE = 5

    # "fee"       = all DB students are bus users (use this if pickup_id not set)
    # "pickup_id" = only students with pickup_id have bus access
    BUS_CHECK_MODE = "fee"

    STUDENT_CACHE_TTL = 300

    INSIGHT_CTX = -1
    DET_SIZE    = (640, 640)

    RESULT_HOLD_SECS  = 8
    VALID_STORE_MAX   = 500
    UNPAID_STORE_MAX  = 500
    INVALID_STORE_MAX = 500
    NOT_UNI_STORE_MAX = 200

    THROTTLE_SECS  = 3.0
    LIVE_JPEG_Q    = 55
    CAPTURE_JPEG_Q = 88
    ENROLL_JPEG_Q  = 80


for _d in [Config.DIR_WITH_BUS, Config.DIR_WITHOUT_BUS,
           Config.DIR_INVALID, Config.DIR_PHOTOS, Config.DIR_NOT_UNI]:
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

# Cooldown: gr -> {last_ts, morning:"YYYY-MM-DD", evening:"YYYY-MM-DD"}
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
_upload_sem     = threading.Semaphore(1)
_save_q         = queue.Queue(maxsize=100)


# =============================================================================
# HELPERS
# =============================================================================
def _ms(t0):
    return f"{(time.time()-t0)*1000:.0f}ms"


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _get_slot():
    """morning = before 14:00, evening = 14:00 onwards."""
    return "morning" if datetime.now().hour < 14 else "evening"


def _parse_location(src):
    def _f(k):
        v = src.get(k)
        if v is None:
            return None
        s = str(v).strip()
        return s or None
    def _ff(k):
        v = src.get(k)
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        try:
            return float(s)
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


def _init_db_pool():
    global _db_pool
    try:
        _db_pool = MySQLConnectionPool(
            pool_name="transbuddy", pool_size=Config.DB_POOL_SIZE,
            host=Config.DB_HOST, port=Config.DB_PORT,
            user=Config.DB_USER, password=Config.DB_PASSWORD,
            database=Config.DB_NAME, connect_timeout=5, autocommit=True,
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
    return mysql.connector.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, user=Config.DB_USER,
        password=Config.DB_PASSWORD, database=Config.DB_NAME,
        connect_timeout=5, autocommit=True,
    )


def test_db():
    try:
        c = _get_db()
        cur = c.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) as n FROM students_detail")
        n = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) as n FROM students_detail WHERE fee_status='Paid'")
        paid = cur.fetchone()["n"]
        cur.close(); c.close()
        logger.info(f"DB OK | students={n} fee_paid={paid} bus_mode={Config.BUS_CHECK_MODE}")
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
        if row:
            logger.info(f"DB | gr={gr_no} name={row['student_name']} "
                        f"fee={row['fee_status']} pickup={row['pickup_id']}")
        else:
            logger.warning(f"DB | gr={gr_no} NOT FOUND")
    except Exception as e:
        logger.error(f"DB fetch gr={gr_no}: {e}")
    with _stu_cache_lock:
        _stu_cache[gr_no] = (row, now)
    return row


def invalidate_student(gr_no):
    with _stu_cache_lock:
        _stu_cache.pop(str(gr_no).strip(), None)


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


def precompute_embeddings():
    logger.info(f"Scanning {Config.DIR_PHOTOS}/...")
    base = Path(Config.DIR_PHOTOS)
    store = {}; imgs = {}; ok = 0; fail = 0; by_gr = {}
    for item in sorted(base.iterdir()):
        if item.is_dir():
            gr = item.name.strip()
            if gr:
                ps = sorted(p for p in item.iterdir() if p.suffix.lower() in _IMG_EXT)
                if ps:
                    by_gr.setdefault(gr, []).extend(ps)
        elif item.is_file() and item.suffix.lower() in _IMG_EXT:
            gr = item.stem.strip()
            if gr and gr not in by_gr:
                by_gr[gr] = [item]
    logger.info(f"Found {len(by_gr)} students with photos")
    for gr_no, paths in sorted(by_gr.items()):
        embs = []; thumb = None
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is None: fail += 1; continue
            emb = _emb_from_bgr(bgr)
            if emb is None: fail += 1; continue
            embs.append(emb)
            if thumb is None: thumb = bgr
        if not embs:
            continue
        store[gr_no] = _l2(np.mean(np.stack(embs), axis=0)).reshape(1, -1)
        imgs[gr_no] = _bgr2b64(thumb, Config.ENROLL_JPEG_Q)
        ok += 1
        logger.info(f"  gr={gr_no} {len(embs)} photo(s)")
    with emb_lock:
        embedding_store.clear(); embedding_store.update(store)
        enrollment_imgs.clear(); enrollment_imgs.update(imgs)
    logger.info(f"Embeddings | ok={ok} fail={fail} total={len(store)}")


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
        results.append({
            "embedding": _l2(face.embedding.astype(np.float32)),
            "crop_b64":  _bgr2b64(crop, Config.ENROLL_JPEG_Q),
            "bbox":      [x1, y1, x2, y2],
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
    logger.info(f"Match | best={best_gr} score={best:.4f} margin={margin:.4f}")
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
# ENROLLMENT IMAGE
# =============================================================================
def _enroll_b64(gr):
    with emb_lock:
        b64 = enrollment_imgs.get(str(gr), "")
    if not b64 and gr:
        try:
            candidates = []
            sub = Path(Config.DIR_PHOTOS) / str(gr)
            if sub.is_dir():
                candidates.extend(sorted(sub.glob("*")))
            for ext in (".jpg", ".jpeg", ".png"):
                candidates.append(Path(Config.DIR_PHOTOS) / (str(gr) + ext))
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
# gr -> {last_ts: float, morning: "YYYY-MM-DD", evening: "YYYY-MM-DD"}
# =============================================================================
def _on_cd(gr: str) -> bool:
    """
    Block if already validated in current time slot today.
    Also hard-block rapid re-triggers within 5 minutes.
    """
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
    slot = _get_slot()
    today = _today()
    with cd_lock:
        rec = student_cooldown.get(gr, {})
        rec["last_ts"] = time.time()
        rec[slot] = today
        student_cooldown[gr] = rec
    logger.info(f"Cooldown SET | gr={gr} slot={slot} date={today}")


def _cd_left(gr: str) -> int:
    with cd_lock:
        rec = student_cooldown.get(gr)
        if rec is None:
            return 0
        hard = max(0, int(300 - (time.time() - rec.get("last_ts", 0))))
        if hard > 0:
            return hard
        slot = _get_slot()
        today = _today()
        if rec.get(slot) != today:
            return 0
        now = datetime.now()
        if slot == "morning":
            next_open = now.replace(hour=14, minute=0, second=0, microsecond=0)
        else:
            next_open = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        return max(0, int((next_open - now).total_seconds()))


# =============================================================================
# THROTTLE (for unknown persons — don't spam /not_uni_student)
# =============================================================================
def _throttle_ok(key: str) -> bool:
    now = time.time()
    with _throttle_lock:
        if now - _throttle.get(key, 0.0) < Config.THROTTLE_SECS:
            return False
        _throttle[key] = now
    return True


# =============================================================================
# STORE FUNCTIONS — one per route, never overlap
# =============================================================================
def _store_not_uni(crop_b64, conf, candidate_gr, msg, location=None):
    """Route A"""
    rec = {
        "route": "not_uni_student", "event": "unknown_person",
        "timestamp": datetime.now().isoformat(),
        "confidence": round(float(conf), 4) if conf else None,
        "candidate_gr": candidate_gr, "message": msg,
        "captured_b64": crop_b64, "enrollment_b64": None,
        "location": location or {},
    }
    with nu_lock:
        not_uni_store.insert(0, rec)
        if len(not_uni_store) > Config.NOT_UNI_STORE_MAX:
            not_uni_store.pop()
    logger.warning(f"[A] not_uni | conf={conf:.3f} stop={location and location.get('stop_name')}")


def _store_invalid(reason, crop_b64, conf, gr, msg, stu=None, location=None):
    """Routes B+C"""
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
    }
    with inv_lock:
        invalid_store.insert(0, rec)
        if len(invalid_store) > Config.INVALID_STORE_MAX:
            invalid_store.pop()
    logger.warning(f"[B/C] invalid | reason={reason} gr={gr} stop={location and location.get('stop_name')}")


def _store_unpaid(stu, crop_b64, location=None):
    """Route D"""
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
        "location": location or {},
    }
    with unpaid_lock:
        unpaid_store[gr] = rec
        if len(unpaid_store) > Config.UNPAID_STORE_MAX:
            del unpaid_store[next(iter(unpaid_store))]
    logger.warning(f"[D] unpaid | gr={gr} stop={location and location.get('stop_name')}")


def _store_valid(stu, crop_b64, location=None):
    """Route E"""
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
        "location": location or {},
    }
    with valid_lock:
        valid_store.insert(0, rec)
        if len(valid_store) > Config.VALID_STORE_MAX:
            valid_store.pop()
    logger.info(f"[E] GRANTED | gr={gr} name={stu.get('student_name')} stop={location and location.get('stop_name')}")


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
# DECISION TREE — ONE FACE
# =============================================================================
def process_one_face(emb, crop_b64, frame, location) -> dict:
    ts = datetime.now().isoformat()
    candidate_gr, matched_gr, best, second = match_face(emb)
    margin = best - second

    # ── ROUTE A: No match ─────────────────────────────────────
    if matched_gr is None:
        key = f"not_uni_{candidate_gr or 'unk'}"
        result = {
            "status": "not_uni", "route": "not_uni_student", "category": "not_uni_student",
            "gr_no": None, "enrollment_no": None, "name": "Unknown Person",
            "department": None, "semester": None, "shift": None,
            "fee_status": None, "pickup_id": None,
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": f"No match (score={best:.3f}, margin={margin:.3f})",
            "timestamp": ts, "captured_b64": crop_b64, "enrollment_b64": None,
            "location": location,
        }
        if _throttle_ok(key):
            _save(frame, Config.DIR_NOT_UNI, "not_uni")
            _store_not_uni(crop_b64, best, candidate_gr, result["message"], location)
        return result

    # ── Cooldown ───────────────────────────────────────────────
    if _on_cd(matched_gr):
        return {
            "status": "on_cooldown", "route": "cooldown", "category": "cooldown",
            "gr_no": matched_gr, "enrollment_no": None, "name": None,
            "department": None, "semester": None, "shift": None,
            "fee_status": None, "pickup_id": None, "on_cooldown": True,
            "cooldown_secs": _cd_left(matched_gr),
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": f"GR {matched_gr} on cooldown — next slot in {_cd_left(matched_gr)//3600}h",
            "timestamp": ts, "captured_b64": crop_b64,
            "enrollment_b64": _enroll_b64(matched_gr), "location": location,
        }

    enr_b64 = _enroll_b64(matched_gr)
    stu = fetch_student(matched_gr)

    # ── ROUTE B: Photo matched, not in DB ──────────────────────
    if stu is None:
        msg = f"GR '{matched_gr}' photo matched but NOT in students_detail"
        result = {
            "status": "invalid_database", "route": "invalid_alerts", "category": "not_in_db",
            "gr_no": matched_gr, "enrollment_no": None, "name": None,
            "department": None, "semester": None, "shift": None,
            "fee_status": None, "pickup_id": None,
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": msg, "timestamp": ts, "captured_b64": crop_b64,
            "enrollment_b64": enr_b64, "location": location,
        }
        with _nodb_lock:
            first = matched_gr not in _nodb_reported
            _nodb_reported[matched_gr] = time.time()
        if first:
            _save(frame, Config.DIR_INVALID, f"nodb_{matched_gr}")
            _store_invalid("not_in_db", crop_b64, best, matched_gr, msg, location=location)
        _set_cd(matched_gr)
        return result

    # ── ROUTE C: In DB, no bus ─────────────────────────────────
    if not _has_bus(stu):
        msg = (f"{stu['student_name']} (GR:{matched_gr}) has no bus subscription "
               f"[pickup_id={stu.get('pickup_id')}]")
        result = {
            "status": "invalid_person", "route": "invalid_alerts", "category": "no_bus_policy",
            "gr_no": str(stu["gr_no"]), "enrollment_no": str(stu["enrollment_no"]),
            "name": stu["student_name"], "department": stu["department"],
            "semester": str(stu["semester"]), "shift": stu["shift"],
            "fee_status": stu["fee_status"], "pickup_id": str(stu.get("pickup_id", "")),
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": msg, "timestamp": ts, "captured_b64": crop_b64,
            "enrollment_b64": enr_b64, "location": location,
        }
        _save(frame, Config.DIR_INVALID, f"nobus_{matched_gr}")
        _store_invalid("no_bus_policy", crop_b64, best, matched_gr, msg, stu=stu, location=location)
        _set_cd(matched_gr)
        return result

    # ── ROUTE D: Bus, fee UNPAID ───────────────────────────────
    if not _fee_paid(stu):
        msg = f"Bus user — FEE UNPAID: {stu['student_name']} (GR:{matched_gr})"
        result = {
            "status": "valid_without_bus", "route": "unpaid_students", "category": "unpaid_fee",
            "gr_no": str(stu["gr_no"]), "enrollment_no": str(stu["enrollment_no"]),
            "name": stu["student_name"], "department": stu["department"],
            "semester": str(stu["semester"]), "shift": stu["shift"],
            "fee_status": stu["fee_status"], "pickup_id": str(stu.get("pickup_id", "")),
            "confidence": round(float(best), 4), "margin": round(float(margin), 4),
            "message": msg, "timestamp": ts, "captured_b64": crop_b64,
            "enrollment_b64": enr_b64, "location": location,
        }
        _save(frame, Config.DIR_WITHOUT_BUS, matched_gr)
        _store_unpaid(stu, crop_b64, location=location)
        _set_cd(matched_gr)
        return result

    # ── ROUTE E: Bus + fee PAID → ACCESS GRANTED ──────────────
    msg = f"ACCESS GRANTED — {stu['student_name']} (GR:{matched_gr}, fee PAID)"
    result = {
        "status": "valid_with_bus", "route": "valid_with_bus", "category": "valid_with_bus",
        "gr_no": str(stu["gr_no"]), "enrollment_no": str(stu["enrollment_no"]),
        "name": stu["student_name"], "department": stu["department"],
        "semester": str(stu["semester"]), "shift": stu["shift"],
        "fee_status": stu["fee_status"], "pickup_id": str(stu.get("pickup_id", "")),
        "confidence": round(float(best), 4), "margin": round(float(margin), 4),
        "message": msg, "timestamp": ts, "captured_b64": crop_b64,
        "enrollment_b64": enr_b64, "location": location,
    }
    _save(frame, Config.DIR_WITH_BUS, matched_gr)
    _store_valid(stu, crop_b64, location=location)
    _set_cd(matched_gr)
    return result


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/")
def dashboard():
    return send_from_directory("templates", "index.html")


def _mjpeg_gen():
    while True:
        with _live_lock:
            frame = _live_frame
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, Config.LIVE_JPEG_Q])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.08)


@app.route("/live_feed")
def live_feed():
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live_frame")
def live_frame_route():
    with _live_lock:
        b64 = _live_frame_b64
    return jsonify({"frame": b64, "ready": b64 is not None})


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
                    "pickup_id": int(row["pickup_id"]),
                    "name": str(row["pickup_name"]).strip(),
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                    "city": str(row.get("city", "") or "").strip(),
                    "state": str(row.get("state", "") or "").strip(),
                    "country": str(row.get("country", "") or "").strip(),
                    "display_name": "",
                })
            except:
                continue
        logger.info(f"/pickup_points | {len(stops)} stops")
        return jsonify({"total": len(stops), "stops": stops})
    except Exception as e:
        logger.error(f"/pickup_points: {e}")
        return jsonify({"error": str(e), "total": 0, "stops": []}), 500


@app.route("/bus_location", methods=["GET", "OPTIONS"])
def bus_location():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with _bus_loc_lock:
        loc = copy.deepcopy(_bus_location)

    # Normalize output and ensure coordinates are float or null
    def _num(v):
        if v is None:
            return None
        try:
            return float(v)
        except:
            return None

    out = {
        "gps_lat": _num(loc.get("gps_lat")),
        "gps_lon": _num(loc.get("gps_lon")),
        "stop_name": loc.get("stop_name") or None,
        "stop_lat": _num(loc.get("stop_lat")),
        "stop_lon": _num(loc.get("stop_lon")),
        "stop_city": loc.get("stop_city") or None,
        "stop_state": loc.get("stop_state") or None,
        "stop_country": loc.get("stop_country") or None,
        "stop_display": loc.get("stop_display") or None,
        "image_quality": _num(loc.get("image_quality")),
        "updated_at": loc.get("updated_at") or None,
    }

    return jsonify(out)


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    t0 = time.time()
    if not _upload_sem.acquire(blocking=False):
        return jsonify({"status": "busy", "face_count": 0, "results": [],
                        "summary": {"valid_with_bus": 0, "unpaid": 0, "invalid": 0, "not_uni": 0, "cooldown": 0},
                        "message": "Server busy"})
    try:
        location = _parse_location(request.form)
        _update_bus_location(location)
        logger.info(f"Upload | stop='{location.get('stop_name', '?')}' "
                    f"gps=({location.get('gps_lat')},{location.get('gps_lon')})")

        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No image"}), 400
        raw = request.files["image"].read()
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "error", "message": "Cannot decode image"}), 400

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, Config.LIVE_JPEG_Q])
        frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        with _live_lock:
            global _live_frame, _live_frame_b64
            _live_frame = frame
            _live_frame_b64 = frame_b64

        detected = detect_all_faces(frame)
        logger.info(f"Upload | {len(detected)} face(s) | {_ms(t0)}")

        if not detected:
            return jsonify({
                "status": "no_face", "face_count": 0, "results": [], "is_idle": False,
                "summary": {"valid_with_bus": 0, "unpaid": 0, "invalid": 0, "not_uni": 0, "cooldown": 0},
                "location": location, "message": "No face detected",
                "timestamp": datetime.now().isoformat(), "captured_b64": frame_b64,
            })

        results = []
        for idx, fi in enumerate(detected):
            r = process_one_face(fi["embedding"], fi["crop_b64"], frame, location)
            r["face_index"] = idx
            r["bbox"] = fi["bbox"]
            results.append(r)
            logger.info(f"  Face {idx+1} | route={r.get('route')} status={r['status']} "
                        f"gr={r.get('gr_no')} name={r.get('name')}")

        n = len(results)
        summary = {
            "valid_with_bus": sum(1 for r in results if r["status"] == "valid_with_bus"),
            "unpaid":         sum(1 for r in results if r["status"] == "valid_without_bus"),
            "invalid":        sum(1 for r in results if r["status"] in ("invalid_person", "invalid_database")),
            "not_uni":        sum(1 for r in results if r["status"] == "not_uni"),
            "cooldown":       sum(1 for r in results if r["status"] == "on_cooldown"),
        }
        scan = {
            "is_idle": False, "multi_face": n > 1, "face_count": n,
            "results": results, "summary": summary, "location": location,
            "timestamp": datetime.now().isoformat(), "captured_b64": frame_b64,
        }
        _push_scan(scan)
        logger.info(f"Done | faces={n} granted={summary['valid_with_bus']} "
                    f"unpaid={summary['unpaid']} invalid={summary['invalid']} "
                    f"not_uni={summary['not_uni']} | {_ms(t0)}")
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
    """Route E — access granted log"""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:    limit = int(request.args.get("limit", 100))
    except: limit = 100
    with valid_lock:
        data = valid_store[:limit]
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
    """Route D — unpaid bus students"""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    gr = request.args.get("gr_no", "").strip()
    with unpaid_lock:
        data = ([unpaid_store[gr]] if gr in unpaid_store else []) if gr else list(unpaid_store.values())
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
    """Routes B+C — not_in_db and no_bus_policy"""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    reason = request.args.get("reason", "").strip()
    try:    limit = int(request.args.get("limit", Config.INVALID_STORE_MAX))
    except: limit = Config.INVALID_STORE_MAX
    with inv_lock:
        data = [r for r in invalid_store if r.get("reason") == reason] if reason else list(invalid_store)
        data = data[:limit]
        if request.args.get("clear", "0") == "1":
            if reason: invalid_store[:] = [r for r in invalid_store if r.get("reason") != reason]
            else:      invalid_store.clear()
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
    """Route A — unknown persons"""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:    limit = int(request.args.get("limit", Config.NOT_UNI_STORE_MAX))
    except: limit = Config.NOT_UNI_STORE_MAX
    with nu_lock:
        data = not_uni_store[:limit]
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


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with emb_lock:
        n = len(embedding_store); grs = sorted(embedding_store.keys())
    with _bus_loc_lock:
        bl = copy.deepcopy(_bus_location)
    return jsonify({
        "status": "ok", "version": "11.0.0",
        "model_loaded": face_app is not None,
        "students_loaded": n, "gr_numbers": grs,
        "bus_check_mode": Config.BUS_CHECK_MODE,
        "bus_location": bl, "timestamp": datetime.now().isoformat(),
    })


@app.route("/debug", methods=["GET", "OPTIONS"])
def debug():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with emb_lock:
        loaded = sorted(embedding_store.keys())
    db_ok = False; db_n = 0; db_paid = 0; sample = []
    try:
        c = _get_db(); cur = c.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) as n FROM students_detail")
        db_n = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) as n FROM students_detail WHERE fee_status='Paid'")
        db_paid = cur.fetchone()["n"]
        cur.execute("SELECT gr_no,student_name,fee_status,pickup_id FROM students_detail LIMIT 5")
        sample = cur.fetchall(); cur.close(); c.close(); db_ok = True
    except Exception as e:
        sample = [{"error": str(e)}]
    today = _today()
    with cd_lock:
        cds = {
            gr: {"morning": rec.get("morning") == today, "evening": rec.get("evening") == today,
                 "next_secs": _cd_left(gr)}
            for gr, rec in student_cooldown.items()
            if rec.get("morning") == today or rec.get("evening") == today
        }
    with inv_lock:   inv_n = len(invalid_store)
    with nu_lock:    nu_n  = len(not_uni_store)
    with unpaid_lock: up_n = len(unpaid_store)
    with valid_lock:  vl_n = len(valid_store)
    with scan_lock:   idle = latest_scan.get("is_idle", True)
    with _bus_loc_lock: bl = copy.deepcopy(_bus_location)
    return jsonify({
        "version": "11.0.0",
        "bus_check_mode": Config.BUS_CHECK_MODE,
        "confidence": Config.CONFIDENCE_THRESHOLD,
        "margin": Config.MARGIN_THRESHOLD,
        "daily_slot_system": {"today": today, "current_slot": _get_slot(), "max_per_day": 2},
        "embeddings": loaded,
        "model_loaded": face_app is not None,
        "db": {"connected": db_ok, "total": db_n, "fee_paid": db_paid, "sample": sample},
        "route_counts": {
            "A_not_uni": nu_n, "B_C_invalid": inv_n, "D_unpaid": up_n, "E_valid": vl_n
        },
        "active_cooldowns_today": cds,
        "live_bus_location": bl,
        "latest_scan_is_idle": idle,
        "decision_tree": {
            "A": "No photo match    -> /not_uni_student",
            "B": "Photo, no DB      -> /invalid_alerts (not_in_db)",
            "C": "In DB, no bus     -> /invalid_alerts (no_bus_policy)",
            "D": "Bus, fee UNPAID   -> /unpaid_students",
            "E": "Bus, fee PAID     -> /valid_students (ACCESS GRANTED)",
        },
        "endpoints": {
            "POST /upload": "Pi sends image+GPS",
            "GET /upload_status": "Latest scan",
            "GET /bus_location": "Live GPS",
            "GET /pickup_points": "Stops from DB",
            "GET /valid_students": "Route E — granted",
            "GET /unpaid_students": "Route D — unpaid",
            "GET /invalid_alerts": "Route B+C",
            "GET /not_uni_student": "Route A — unknown",
            "GET /cooldown_status": "Daily slots",
            "GET /health": "Health",
            "GET /debug": "This page",
        },
    })


# =============================================================================
# STARTUP
# =============================================================================
def startup():
    global face_app
    logger.info("=" * 64)
    logger.info("  TransBuddy Server v11.0.0 — Marwadi University")
    logger.info("=" * 64)
    logger.info(f"  DB          : {Config.DB_HOST}/{Config.DB_NAME}")
    logger.info(f"  Photos      : {Config.DIR_PHOTOS}/")
    logger.info(f"  Confidence  : {Config.CONFIDENCE_THRESHOLD}")
    logger.info(f"  Margin      : {Config.MARGIN_THRESHOLD}")
    logger.info(f"  Bus mode    : {Config.BUS_CHECK_MODE}")
    logger.info(f"  Daily slots : morning (before 14:00) + evening (14:00+)")
    logger.info("=" * 64)
    logger.info("  ROUTES (per face — strict single route):")
    logger.info("    A. No photo match  -> /not_uni_student")
    logger.info("    B. Photo, no DB    -> /invalid_alerts (not_in_db)")
    logger.info("    C. In DB, no bus   -> /invalid_alerts (no_bus_policy)")
    logger.info("    D. Bus, UNPAID     -> /unpaid_students")
    logger.info("    E. Bus, PAID       -> /valid_students (GRANTED)")
    logger.info("=" * 64)
    _init_db_pool()
    test_db()
    face_app = _init_model()
    precompute_embeddings()
    threading.Thread(target=_save_worker, daemon=True, name="save").start()
    logger.info("  http://0.0.0.0:5000")
    logger.info("  http://localhost:5000/debug")
    logger.info("=" * 64)


if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
