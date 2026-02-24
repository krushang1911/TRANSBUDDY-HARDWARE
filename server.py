"""
TransBuddy Bus Face Verification Server
Marwadi University - Production System
Version: 5.0.0

Database: transbuddy_db_1
Table:    students_detail
Columns:  gr_no | enrollment_no | student_name | department | semester | shift | fee_status | created_at | pickup_id

Enrollment photos folder (FLAT):
    photos/
        123014.jpg   <- filename = gr_no exactly
        134502.jpg

Features:
    1. ArcFace (InsightFace) face recognition with vectorized matching
    2. Per-student cooldown (10 min) - same student skipped for STUDENT_COOLDOWN_SECS
    3. Unpaid students stored on YOUR server - friend fetches via GET /unpaid_students
    4. Invalid alerts stored in memory - view via GET /invalid_alerts
    5. Live camera feed via base64 polling (works through ngrok)
    6. Dashboard result priority-hold (6 sec)
    7. Full debug endpoint at /debug
    8. All ports open: 0.0.0.0:5000 with custom CORS headers (no flask-cors dependency)
"""

import os
import base64
import time
import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import mysql.connector
from flask import Flask, request, jsonify, send_from_directory, Response
from insightface.app import FaceAnalysis

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION  <-- Edit settings here
# =============================================================================
class Config:
    # Face matching
    CONFIDENCE_THRESHOLD  = 0.55   # raise to 0.65 after good quality photos enrolled
    MARGIN_THRESHOLD      = 0.05   # min gap between best and 2nd best match score

    # Per-student cooldown: after any result, same student ignored for this long
    STUDENT_COOLDOWN_SECS = 600    # 10 minutes

    # Capture save folders
    DIR_WITH_BUS    = "captures/with_bus"
    DIR_WITHOUT_BUS = "captures/without_bus"
    DIR_INVALID     = "captures/invalid_captures"

    # Enrollment photos folder — flat structure: photos/GR_NO.jpg
    DIR_PHOTOS = "photos"

    # MySQL database
    DB_HOST     = "localhost"
    DB_PORT     = 3306
    DB_USER     = "root"
    DB_PASSWORD = ""        # set your MySQL password if any
    DB_NAME     = "transbuddy_db_1"

    # InsightFace model
    INSIGHT_CTX = -1        # -1 = CPU, 0 = GPU

    # Dashboard result hold time
    RESULT_HOLD_SECS = 6    # hold result on dashboard for this many seconds

    # Unpaid students API key — leave "" for open access
    UNPAID_API_KEY = ""

    # Invalid alerts — keep last N in memory
    INVALID_STORE_MAX = 100


# =============================================================================
# CREATE REQUIRED DIRECTORIES
# =============================================================================
for _d in [Config.DIR_WITH_BUS, Config.DIR_WITHOUT_BUS,
           Config.DIR_INVALID,   Config.DIR_PHOTOS]:
    Path(_d).mkdir(parents=True, exist_ok=True)


# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__, static_folder="static", template_folder="templates")


@app.before_request
def handle_preflight():
    """
    Handle OPTIONS preflight requests with open headers.
    """
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"]      = "*"
        response.headers["Access-Control-Allow-Methods"]     = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
        response.headers["Access-Control-Allow-Headers"]     = "*"
        response.headers["Access-Control-Max-Age"]           = "3600"
        return response


@app.after_request
def add_open_headers(response):
    """
    Add headers to all responses for maximum openness.
    """
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Cache-Control"]                = "no-cache, no-store, must-revalidate"
    return response


# =============================================================================
# GLOBAL STATE
# =============================================================================
face_app        = None
embedding_store = {}    # { "123014": np.ndarray shape (1,512) }
enrollment_imgs = {}    # { "123014": "data:image/jpeg;base64,..." }
embedding_lock  = threading.Lock()

latest_result = {
    "status":         "idle",
    "gr_no":          None,
    "enrollment_no":  None,
    "name":           None,
    "department":     None,
    "semester":       None,
    "shift":          None,
    "fee_status":     None,
    "confidence":     None,
    "message":        None,
    "timestamp":      None,
    "captured_b64":   None,
    "enrollment_b64": None,
}
result_lock     = threading.Lock()
_last_push_time = 0.0

# Per-student cooldown: { "123014": 1708950000.0 }
student_cooldown = {}
cooldown_lock    = threading.Lock()

# Unpaid students store: { "123014": { ...student details + enrollment_b64 } }
unpaid_store      = {}
unpaid_store_lock = threading.Lock()

# Invalid alerts store — last 100, newest first
invalid_store      = []
invalid_store_lock = threading.Lock()

# Latest Pi camera frame for live feed
latest_frame      = None
latest_frame_lock = threading.Lock()

# Priority for dashboard result hold logic
_PRIORITY = {
    "valid_with_bus":    4,
    "valid_without_bus": 3,
    "invalid_database":  2,
    "invalid_person":    1,
    "idle":              0,
}


# =============================================================================
# COOLDOWN HELPERS
# =============================================================================
def is_student_on_cooldown(gr_no: str) -> bool:
    """True if student was processed recently and should be skipped."""
    with cooldown_lock:
        last = student_cooldown.get(gr_no)
        if last is None:
            return False
        return (time.time() - last) < Config.STUDENT_COOLDOWN_SECS


def set_student_cooldown(gr_no: str):
    """Start cooldown timer for this student."""
    with cooldown_lock:
        student_cooldown[gr_no] = time.time()
    logger.info(f"Cooldown SET | gr={gr_no} | next scan in {Config.STUDENT_COOLDOWN_SECS}s")


def get_cooldown_remaining(gr_no: str) -> int:
    """Seconds left in cooldown, 0 if not on cooldown."""
    with cooldown_lock:
        last = student_cooldown.get(gr_no)
        if last is None:
            return 0
        return max(0, int(Config.STUDENT_COOLDOWN_SECS - (time.time() - last)))


# =============================================================================
# DATABASE
# =============================================================================
def get_db():
    return mysql.connector.connect(
        host            = Config.DB_HOST,
        port            = Config.DB_PORT,
        user            = Config.DB_USER,
        password        = Config.DB_PASSWORD,
        database        = Config.DB_NAME,
        connect_timeout = 5,
        autocommit      = True,
    )


def test_db_connection():
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students_detail")
        count  = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        logger.info(f"Database OK | students_detail has {count} records")
        return True
    except mysql.connector.Error as e:
        logger.error(f"Database FAILED | {e}")
        return False


def fetch_student(gr_no: str) -> dict | None:
    """Fetch one student by gr_no. Returns dict or None."""
    try:
        conn   = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT gr_no, enrollment_no, student_name,
                   department, semester, shift, fee_status
            FROM   students_detail
            WHERE  gr_no = %s
            LIMIT  1
            """,
            (str(gr_no).strip(),)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            logger.info(f"DB OK | gr={row['gr_no']} name={row['student_name']} fee={row['fee_status']}")
        else:
            logger.warning(f"DB: gr_no='{gr_no}' NOT FOUND in students_detail")
        return row
    except mysql.connector.Error as e:
        logger.error(f"DB error gr='{gr_no}': {e}")
        return None


# =============================================================================
# MODEL
# =============================================================================
def init_model() -> FaceAnalysis:
    logger.info("Loading InsightFace ArcFace model (buffalo_l) ...")
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=Config.INSIGHT_CTX, det_size=(640, 640))
    logger.info("Model loaded OK")
    return fa


# =============================================================================
# EMBEDDING UTILITIES
# =============================================================================
def l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)


def extract_embedding(bgr: np.ndarray) -> np.ndarray | None:
    faces = face_app.get(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return l2_normalize(face.embedding.astype(np.float32))


def bgr_to_b64(bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# =============================================================================
# PRECOMPUTE EMBEDDINGS FROM photos/ FOLDER
# =============================================================================
def precompute_embeddings():
    """
    Scan photos/ folder. Each filename stem = gr_no.
    photos/123014.jpg  ->  gr_no = "123014"
    """
    logger.info(f"Scanning '{Config.DIR_PHOTOS}/' ...")
    store   = {}
    enr_img = {}
    base    = Path(Config.DIR_PHOTOS)
    total   = 0
    failed  = 0

    files = [p for p in sorted(base.glob("*"))
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    logger.info(f"Found {len(files)} photo(s)")

    for img_path in files:
        gr_no = img_path.stem.strip()
        if not gr_no:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Cannot read: {img_path}")
            failed += 1
            continue
        emb = extract_embedding(img)
        if emb is None:
            logger.warning(f"No face in {img_path.name} | gr='{gr_no}'")
            failed += 1
            continue
        store[gr_no]   = emb.reshape(1, -1)
        enr_img[gr_no] = bgr_to_b64(img)
        logger.info(f"  Loaded: {img_path.name} -> gr='{gr_no}'")
        total += 1

    with embedding_lock:
        embedding_store.clear()
        embedding_store.update(store)
        enrollment_imgs.clear()
        enrollment_imgs.update(enr_img)

    logger.info(f"Embeddings ready | Loaded:{total} Failed:{failed} GRs:{sorted(store.keys())}")


# =============================================================================
# MATCHING
# =============================================================================
def match_embedding(live_emb: np.ndarray):
    """
    Vectorized cosine similarity vs all stored embeddings.
    Returns (best_candidate_gr, matched_gr | None, best_score, second_score).
    """
    with embedding_lock:
        if not embedding_store:
            logger.warning("No embeddings! Add photos to photos/ folder.")
            return None, None, 0.0, 0.0
        scores = {gr: float((mat @ live_emb).max())
                  for gr, mat in embedding_store.items()}

    ranked        = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_gr, best = ranked[0]
    second        = ranked[1][1] if len(ranked) > 1 else 0.0
    margin        = best - second

    logger.info(
        f"Match | best_gr='{best_gr}' score={best:.3f} margin={margin:.3f} "
        f"(need >={Config.CONFIDENCE_THRESHOLD} margin>={Config.MARGIN_THRESHOLD})"
    )

    if best >= Config.CONFIDENCE_THRESHOLD and margin >= Config.MARGIN_THRESHOLD:
        return best_gr, best_gr, best, second
    return best_gr, None, best, second


# =============================================================================
# FEE CHECK
# =============================================================================
def is_fee_paid(fee_status: str) -> bool:
    """Case-insensitive. DB stores 'Paid' or 'Unpaid'."""
    return str(fee_status).strip().lower() == "paid"


# =============================================================================
# SAVE CAPTURE TO DISK
# =============================================================================
def save_capture(bgr: np.ndarray, folder: str, label: str = "unknown") -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(folder, f"{label}_{ts}.jpg")
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


# =============================================================================
# STORE INVALID ALERT
# =============================================================================
def store_invalid_alert(reason: str, captured_b64: str,
                        confidence=None, gr_no=None, message=""):
    """
    Save every invalid event to memory. Last 100 kept, newest first.
    reason values:
        "no_face"    - frame received but no human face detected
        "unknown"    - face found but no match above threshold
        "not_in_db"  - face matched a photo but GR not found in DB
    """
    gr = str(gr_no).strip() if gr_no is not None else ""

    student = None
    if gr:
        try:
            student = fetch_student(gr)
        except Exception:
            student = None

    with embedding_lock:
        enr_photo_b64 = enrollment_imgs.get(gr, "")

    record = {
        "reason":         reason,
        "timestamp":      datetime.now().isoformat(),
        "confidence":     round(float(confidence), 4) if confidence is not None else None,
        "gr_no":          gr if gr else None,
        "enrollment_no":  (student.get("enrollment_no") if student else None),
        "student_name":   (student.get("student_name") if student else None),
        "department":     (student.get("department") if student else None),
        "semester":       (str(student.get("semester")) if student and student.get("semester") is not None else None),
        "shift":          (student.get("shift") if student else None),
        "fee_status":     (student.get("fee_status") if student else None),
        "message":        message,
        "captured_b64":   captured_b64,
        "enrollment_b64": enr_photo_b64,
    }
    with invalid_store_lock:
        invalid_store.insert(0, record)
        if len(invalid_store) > Config.INVALID_STORE_MAX:
            invalid_store.pop()
    logger.warning(
        f"INVALID ALERT | reason={reason} gr={gr_no} "
        f"conf={confidence} total={len(invalid_store)}"
    )


# =============================================================================
# STORE UNPAID STUDENT
# =============================================================================
def store_unpaid_student(student: dict):
    """
    Saves unpaid student to memory on YOUR server.
    Friend calls GET /unpaid_students on your ngrok URL to fetch.
    """
    gr_no = str(student.get("gr_no", ""))

    with embedding_lock:
        enr_photo_b64 = enrollment_imgs.get(gr_no, "")

    record = {
        "event":          "unpaid_student_detected",
        "timestamp":      datetime.now().isoformat(),
        "gr_no":          gr_no,
        "enrollment_no":  str(student.get("enrollment_no", "")),
        "student_name":   student.get("student_name", ""),
        "department":     student.get("department", ""),
        "semester":       str(student.get("semester", "")),
        "shift":          student.get("shift", ""),
        "fee_status":     student.get("fee_status", ""),
        "enrollment_b64": enr_photo_b64,
    }

    with unpaid_store_lock:
        unpaid_store[gr_no] = record   # keyed by gr_no — no duplicates

    logger.info(
        f"Unpaid stored | gr={gr_no} name={record['student_name']} "
        f"total_unpaid={len(unpaid_store)}"
    )


# =============================================================================
# LOCAL NOTIFICATION LOG
# =============================================================================
def send_notification(gr_no, reason: str):
    logger.warning(f"NOTIFICATION | gr={gr_no} | reason={reason}")


# =============================================================================
# RESULT PUSH WITH PRIORITY HOLD
# =============================================================================
def _push(result: dict):
    """
    Update latest_result. Holds current result for RESULT_HOLD_SECS.
    During hold, only higher-priority result can overwrite.
    """
    global _last_push_time
    new_p = _PRIORITY.get(result.get("status", "idle"), 0)

    with result_lock:
        cur_p   = _PRIORITY.get(latest_result.get("status", "idle"), 0)
        elapsed = time.time() - _last_push_time
        held    = elapsed < Config.RESULT_HOLD_SECS
        if held and new_p <= cur_p:
            return
        latest_result.update(result)
        _last_push_time = time.time()


def _ms(t0: float) -> str:
    return f"{(time.time()-t0)*1000:.1f}ms"


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def dashboard():
    return send_from_directory("templates", "index.html")


# ── Live MJPEG stream (local use) ─────────────────────────────────────────────
def _mjpeg_gen():
    while True:
        with latest_frame_lock:
            frame = latest_frame
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
        time.sleep(0.1)


@app.route("/live_feed")
def live_feed():
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Live frame as base64 JSON (works through ngrok) ───────────────────────────
@app.route("/live_frame", methods=["GET"])
def live_frame():
    """
    Dashboard polls this every 200ms for live camera feed.
    Returns base64 JPEG — works through ngrok unlike MJPEG.
    """
    with latest_frame_lock:
        frame = latest_frame
    if frame is None:
        return jsonify({"frame": None, "ready": False})
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    return jsonify({"frame": b64, "ready": True})


# ── Main upload endpoint (Raspberry Pi posts frames here) ─────────────────────
@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    # OPTIONS is handled globally by before_request, but kept here as fallback
    if request.method == "OPTIONS":
        return jsonify({}), 200

    t0 = time.time()

    # 1. Decode image from Pi
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image field"}), 400

    raw   = request.files["image"].read()
    nparr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "Cannot decode image"}), 400

    # Update live feed
    with latest_frame_lock:
        global latest_frame
        latest_frame = frame.copy()

    cap_b64 = bgr_to_b64(frame)

    # 2. Extract face embedding
    live_emb = extract_embedding(frame)
    if live_emb is None:
        save_capture(frame, Config.DIR_INVALID, "no_face")
        send_notification(None, "no_face_detected")
        store_invalid_alert(
            reason="no_face",
            captured_b64=cap_b64,
            message="No face detected in frame"
        )
        result = {
            "status":         "invalid_person",
            "gr_no":          None,
            "enrollment_no":  None,
            "name":           None,
            "department":     None,
            "semester":       None,
            "shift":          None,
            "fee_status":     None,
            "confidence":     None,
            "message":        "No face detected in frame",
            "timestamp":      datetime.now().isoformat(),
            "captured_b64":   cap_b64,
            "enrollment_b64": None,
        }
        _push(result)
        logger.info(f"No face | {_ms(t0)}")
        return jsonify(result)

    # 3. Match face against enrollment photos
    candidate_gr, matched_gr, best, second = match_embedding(live_emb)

    if matched_gr is None:
        save_capture(frame, Config.DIR_INVALID, "unknown")
        send_notification(None, f"no_match score={best:.3f}")
        store_invalid_alert(
            reason="unknown",
            captured_b64=cap_b64,
            confidence=best,
            gr_no=candidate_gr,
            message=f"No confident match (score={best:.3f}, margin={best-second:.3f})"
        )
        result = {
            "status":         "invalid_person",
            "gr_no":          None,
            "enrollment_no":  None,
            "name":           None,
            "department":     None,
            "semester":       None,
            "shift":          None,
            "fee_status":     None,
            "confidence":     round(float(best), 4),
            "message":        f"No confident match (score={best:.3f}, margin={best-second:.3f})",
            "timestamp":      datetime.now().isoformat(),
            "captured_b64":   cap_b64,
            "enrollment_b64": None,
        }
        _push(result)
        logger.info(f"No match | score={best:.3f} | {_ms(t0)}")
        return jsonify(result)

    # 4. Per-student cooldown check
    if is_student_on_cooldown(matched_gr):
        remaining = get_cooldown_remaining(matched_gr)
        with embedding_lock:
            enr_b64 = enrollment_imgs.get(matched_gr, "")
        logger.info(f"COOLDOWN SKIP | gr={matched_gr} | {remaining}s left | {_ms(t0)}")
        return jsonify({
            "status":         "on_cooldown",
            "gr_no":          matched_gr,
            "message":        f"Already processed. Cooldown: {remaining}s remaining.",
            "on_cooldown":    True,
            "cooldown_secs":  remaining,
            "confidence":     round(float(best), 4),
            "timestamp":      datetime.now().isoformat(),
            "captured_b64":   cap_b64,
            "enrollment_b64": enr_b64,
        })

    # 5. Get enrollment photo for matched student
    with embedding_lock:
        enr_b64 = enrollment_imgs.get(matched_gr, "")

    # 6. Fetch student from database
    student = fetch_student(matched_gr)

    if student is None:
        save_capture(frame, Config.DIR_INVALID, f"nodb_{matched_gr}")
        send_notification(matched_gr, "not_in_database")
        store_invalid_alert(
            reason="not_in_db",
            captured_b64=cap_b64,
            confidence=best,
            gr_no=matched_gr,
            message=f"GR '{matched_gr}' matched photo but not found in database"
        )
        set_student_cooldown(matched_gr)
        result = {
            "status":         "invalid_database",
            "gr_no":          matched_gr,
            "enrollment_no":  None,
            "name":           None,
            "department":     None,
            "semester":       None,
            "shift":          None,
            "fee_status":     None,
            "confidence":     round(float(best), 4),
            "message":        f"GR '{matched_gr}' matched in photos but not found in database.",
            "timestamp":      datetime.now().isoformat(),
            "captured_b64":   cap_b64,
            "enrollment_b64": enr_b64,
        }
        _push(result)
        logger.warning(f"DB miss | gr='{matched_gr}' score={best:.3f} | {_ms(t0)}")
        return jsonify(result)

    # 7. Fee status routing
    paid = is_fee_paid(student["fee_status"])

    if paid:
        folder = Config.DIR_WITH_BUS
        status = "valid_with_bus"
        msg    = f"Access granted - {student['student_name']} (Bus: allowed)"
    else:
        folder = Config.DIR_WITHOUT_BUS
        status = "valid_without_bus"
        msg    = f"Identity confirmed - {student['student_name']} (Bus: fee unpaid)"
        send_notification(matched_gr, "fee_unpaid")
        store_unpaid_student(student)

    save_capture(frame, folder, matched_gr)

    # 8. Set cooldown
    set_student_cooldown(matched_gr)

    result = {
        "status":         status,
        "gr_no":          str(student["gr_no"]),
        "enrollment_no":  str(student["enrollment_no"]),
        "name":           student["student_name"],
        "department":     student["department"],
        "semester":       str(student["semester"]),
        "shift":          student["shift"],
        "fee_status":     student["fee_status"],
        "confidence":     round(float(best), 4),
        "margin":         round(float(best - second), 4),
        "message":        msg,
        "timestamp":      datetime.now().isoformat(),
        "captured_b64":   cap_b64,
        "enrollment_b64": enr_b64,
    }
    _push(result)
    logger.info(
        f"{status} | gr={matched_gr} name={student['student_name']} "
        f"fee={student['fee_status']} score={best:.3f} | {_ms(t0)}"
    )
    return jsonify(result)


# ── Dashboard status poll ─────────────────────────────────────────────────────
@app.route("/upload_status", methods=["GET", "OPTIONS"])
def upload_status():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with result_lock:
        return jsonify(latest_result)


# ── Cooldown status ───────────────────────────────────────────────────────────
@app.route("/cooldown_status", methods=["GET", "OPTIONS"])
def cooldown_status():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    now = time.time()
    with cooldown_lock:
        active = {
            gr: {
                "remaining_secs": max(0, int(Config.STUDENT_COOLDOWN_SECS - (now - ts))),
                "last_seen":      datetime.fromtimestamp(ts).isoformat(),
            }
            for gr, ts in student_cooldown.items()
            if now - ts < Config.STUDENT_COOLDOWN_SECS
        }
    return jsonify({
        "cooldown_secs_configured": Config.STUDENT_COOLDOWN_SECS,
        "students_on_cooldown":     len(active),
        "active_cooldowns":         active,
    })


# ── Unpaid students endpoint (friend fetches from here) ───────────────────────
@app.route("/unpaid_students", methods=["GET", "OPTIONS"])
def unpaid_students():
    """
    Friend calls this to get all unpaid students detected.
    GET https://YOUR-NGROK-URL.ngrok-free.app/unpaid_students

    Optional query params:
        ?key=SECRET    if you set Config.UNPAID_API_KEY
        ?gr_no=123014  fetch single student
        ?clear=1       clear list after fetching
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if Config.UNPAID_API_KEY:
        if request.args.get("key", "") != Config.UNPAID_API_KEY:
            return jsonify({"error": "Unauthorized. Pass ?key=YOUR_KEY"}), 401

    gr_filter = request.args.get("gr_no", "").strip()

    with unpaid_store_lock:
        if gr_filter:
            data = [unpaid_store[gr_filter]] if gr_filter in unpaid_store else []
        else:
            data = list(unpaid_store.values())

        if request.args.get("clear", "0") == "1":
            if gr_filter:
                unpaid_store.pop(gr_filter, None)
            else:
                unpaid_store.clear()
            logger.info(f"Unpaid store cleared | gr_filter='{gr_filter}'")

    data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    logger.info(f"Fetched /unpaid_students | total={len(data)}")
    return jsonify({"total": len(data), "unpaid_students": data})


# ── Clear unpaid list ─────────────────────────────────────────────────────────
@app.route("/unpaid_students/clear", methods=["POST", "OPTIONS"])
def clear_unpaid():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with unpaid_store_lock:
        count = len(unpaid_store)
        unpaid_store.clear()
    logger.info(f"Unpaid store cleared | {count} records removed")
    return jsonify({"status": "cleared", "removed": count})


# ── Invalid alerts endpoint ───────────────────────────────────────────────────
@app.route("/invalid_alerts", methods=["GET", "OPTIONS"])
def invalid_alerts():
    """
    View all invalid/unknown scan attempts with captured photos.
    http://localhost:5000/invalid_alerts

    Alert types:
        no_face    - no human face detected in frame
        unknown    - face found but no match above threshold
        not_in_db  - face matched a photo but GR not found in DB

    Query params:
        ?reason=unknown     filter by type
        ?limit=20           limit results (default: all)
        ?clear=1            clear after fetching
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    reason_filter = request.args.get("reason", "").strip()
    try:
        limit = int(request.args.get("limit", Config.INVALID_STORE_MAX))
    except ValueError:
        limit = Config.INVALID_STORE_MAX

    with invalid_store_lock:
        if reason_filter:
            data = [r for r in invalid_store if r.get("reason") == reason_filter]
        else:
            data = list(invalid_store)

        data = data[:limit]

        if request.args.get("clear", "0") == "1":
            invalid_store.clear()
            logger.info("Invalid alerts cleared via API")

        counts = {"no_face": 0, "unknown": 0, "not_in_db": 0}
        for r in invalid_store:
            k = r.get("reason", "")
            if k in counts:
                counts[k] += 1

    return jsonify({
        "total":          len(invalid_store),
        "counts_by_type": counts,
        "showing":        len(data),
        "alerts":         data,
    })


# ── Clear invalid alerts ──────────────────────────────────────────────────────
@app.route("/invalid_alerts/clear", methods=["POST", "OPTIONS"])
def clear_invalid_alerts():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with invalid_store_lock:
        count = len(invalid_store)
        invalid_store.clear()
    logger.info(f"Invalid alerts cleared | {count} removed")
    return jsonify({"status": "cleared", "removed": count})


# ── Notification endpoint ─────────────────────────────────────────────────────
@app.route("/notification", methods=["POST", "OPTIONS"])
def notification():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data = request.get_json(silent=True) or {}
    send_notification(data.get("gr_no"), data.get("reason", "unknown"))
    return jsonify({"status": "sent"})


# ── Hot-reload embeddings ─────────────────────────────────────────────────────
@app.route("/reload_embeddings", methods=["POST", "OPTIONS"])
def reload_embeddings():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    secret = os.environ.get("RELOAD_SECRET", "changeme")
    if request.headers.get("X-Reload-Secret", "") != secret:
        return jsonify({"error": "Forbidden"}), 403
    threading.Thread(target=precompute_embeddings, daemon=True).start()
    return jsonify({"status": "reloading"})


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    with embedding_lock:
        n   = len(embedding_store)
        grs = sorted(embedding_store.keys())
    return jsonify({
        "status":          "ok",
        "students_loaded": n,
        "gr_numbers":      grs,
        "timestamp":       datetime.now().isoformat(),
    })


# ── Debug endpoint ────────────────────────────────────────────────────────────
@app.route("/debug", methods=["GET", "OPTIONS"])
def debug():
    """Full system status — http://localhost:5000/debug"""
    if request.method == "OPTIONS":
        return jsonify({}), 200

    base        = Path(Config.DIR_PHOTOS)
    photo_files = [p.name for p in sorted(base.glob("*"))
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    with embedding_lock:
        loaded_grs = sorted(embedding_store.keys())

    db_ok = False; db_count = 0; db_sample = []
    try:
        conn   = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as cnt FROM students_detail")
        db_count = cursor.fetchone()["cnt"]
        cursor.execute("SELECT gr_no, student_name, fee_status FROM students_detail LIMIT 5")
        db_sample = cursor.fetchall()
        cursor.close(); conn.close()
        db_ok = True
    except Exception as e:
        db_sample = [{"error": str(e)}]

    now = time.time()
    with cooldown_lock:
        active_cd = {
            gr: int(Config.STUDENT_COOLDOWN_SECS - (now - ts))
            for gr, ts in student_cooldown.items()
            if now - ts < Config.STUDENT_COOLDOWN_SECS
        }

    with unpaid_store_lock:
        unpaid_count = len(unpaid_store)
        unpaid_grs   = list(unpaid_store.keys())

    with invalid_store_lock:
        invalid_count  = len(invalid_store)
        invalid_counts = {"no_face": 0, "unknown": 0, "not_in_db": 0}
        for r in invalid_store:
            k = r.get("reason", "")
            if k in invalid_counts:
                invalid_counts[k] += 1

    return jsonify({
        "photos_folder":            Config.DIR_PHOTOS,
        "photo_files_found":        photo_files,
        "embeddings_loaded":        loaded_grs,
        "db_connected":             db_ok,
        "db_total_students":        db_count,
        "db_sample_records":        db_sample,
        "config_threshold":         Config.CONFIDENCE_THRESHOLD,
        "config_margin":            Config.MARGIN_THRESHOLD,
        "cooldown_secs":            Config.STUDENT_COOLDOWN_SECS,
        "active_cooldowns":         active_cd,
        "unpaid_stored_count":      unpaid_count,
        "unpaid_gr_numbers":        unpaid_grs,
        "unpaid_fetch_url":         "GET /unpaid_students",
        "invalid_alerts_count":     invalid_count,
        "invalid_alerts_by_type":   invalid_counts,
        "invalid_alerts_fetch_url": "GET /invalid_alerts",
        "cors":                     "enabled for all origins",
        "tip":                      "photo filename (no ext) must exactly match gr_no in DB",
    })


# =============================================================================
# STARTUP
# =============================================================================
def startup():
    global face_app
    logger.info("=" * 60)
    logger.info("  TransBuddy Server v5.0 - Marwadi University")
    logger.info("=" * 60)

    test_db_connection()
    face_app = init_model()
    precompute_embeddings()

    logger.info(f"Student cooldown : {Config.STUDENT_COOLDOWN_SECS}s per student")
    logger.info(f"Invalid max store: {Config.INVALID_STORE_MAX} alerts")
    logger.info(f"Server ports     : ALL open (0.0.0.0:5000)")
    logger.info(f"CORS headers     : Enabled for all origins")
    logger.info("=" * 60)
    logger.info("  Dashboard     : http://localhost:5000")
    logger.info("  Debug         : http://localhost:5000/debug")
    logger.info("  Health        : http://localhost:5000/health")
    logger.info("  Cooldowns     : http://localhost:5000/cooldown_status")
    logger.info("  Invalid alerts: http://localhost:5000/invalid_alerts")
    logger.info("  Unpaid list   : http://localhost:5000/unpaid_students")
    logger.info("=" * 60)


if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
