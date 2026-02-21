"""
TransBuddy Bus Face Verification Server
Marwadi University - Production System
Version: 4.0.0

Database: transbuddy_db_1
Table:    students_detail
Columns:  gr_no | enrollment_no | student_name | department | semester | shift | fee_status | created_at | pickup_id

Enrollment photos folder (FLAT):
    photos/
        123014.jpg      ← filename = gr_no exactly
        134502.jpg
        ...
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

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← Edit DB settings here
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # ── Face matching thresholds ──────────────────────────────────────────────
    CONFIDENCE_THRESHOLD = 0.55   # Lower = more lenient (raise to 0.65 after good photos)
    MARGIN_THRESHOLD     = 0.05   # Min gap between best and 2nd best match

    # ── Capture save folders ──────────────────────────────────────────────────
    DIR_WITH_BUS    = "captures/with_bus"
    DIR_WITHOUT_BUS = "captures/without_bus"
    DIR_INVALID     = "captures/invalid_captures"

    # ── Enrollment photos folder (FLAT: photos/GR_NO.jpg) ────────────────────
    DIR_PHOTOS = "photos"

    # ── MySQL database ────────────────────────────────────────────────────────
    DB_HOST     = "localhost"
    DB_PORT     = 3306
    DB_USER     = "root"
    DB_PASSWORD = ""           # ← Set your MySQL password if any
    DB_NAME     = "transbuddy_db_1"

    # ── InsightFace ───────────────────────────────────────────────────────────
    INSIGHT_CTX = -1           # -1 = CPU, 0 = GPU

    # ── Result hold time (seconds) ────────────────────────────────────────────
    RESULT_HOLD_SECS = 6       # Don't overwrite a result for this many seconds


# ─────────────────────────────────────────────────────────────────────────────
# CREATE REQUIRED DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
for _d in [Config.DIR_WITH_BUS, Config.DIR_WITHOUT_BUS,
           Config.DIR_INVALID,   Config.DIR_PHOTOS]:
    Path(_d).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
face_app        = None
embedding_store = {}   # { "123014": np.ndarray (1, 512) }
enrollment_imgs = {}   # { "123014": "data:image/jpeg;base64,..." }
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

# Latest Pi frame for MJPEG live stream
latest_frame      = None
latest_frame_lock = threading.Lock()

# Priority map for result hold logic
_PRIORITY = {
    "valid_with_bus":    4,
    "valid_without_bus": 3,
    "invalid_database":  2,
    "invalid_person":    1,
    "idle":              0,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE  ── matches your exact table structure
# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    """Create and return a new MySQL connection."""
    return mysql.connector.connect(
        host         = Config.DB_HOST,
        port         = Config.DB_PORT,
        user         = Config.DB_USER,
        password     = Config.DB_PASSWORD,
        database     = Config.DB_NAME,
        connect_timeout = 5,
        autocommit   = True,
    )


def test_db_connection():
    """Test DB at startup and log result."""
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students_detail")
        count  = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        logger.info(f"Database connected ✅ | students_detail has {count} records")
        return True
    except mysql.connector.Error as e:
        logger.error(f"Database connection FAILED ❌ | {e}")
        logger.error("Check: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME in Config")
        return False


def fetch_student(gr_no: str) -> dict | None:
    """
    Fetch student by gr_no from students_detail table.

    Actual columns used:
        gr_no, enrollment_no, student_name, department,
        semester, shift, fee_status
    """
    try:
        conn   = get_db()
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT
                gr_no,
                enrollment_no,
                student_name,
                department,
                semester,
                shift,
                fee_status
            FROM students_detail
            WHERE gr_no = %s
            LIMIT 1
        """
        cursor.execute(query, (str(gr_no).strip(),))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            logger.info(
                f"DB fetch OK | gr={row['gr_no']} | "
                f"name={row['student_name']} | "
                f"fee={row['fee_status']}"
            )
        else:
            logger.warning(f"DB fetch: gr_no='{gr_no}' NOT FOUND in students_detail")

        return row

    except mysql.connector.Error as e:
        logger.error(f"DB error for gr_no='{gr_no}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def init_model() -> FaceAnalysis:
    logger.info("Loading InsightFace ArcFace model (buffalo_l) …")
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=Config.INSIGHT_CTX, det_size=(640, 640))
    logger.info("Model loaded ✅")
    return fa


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE EMBEDDINGS FROM photos/ FOLDER
# ─────────────────────────────────────────────────────────────────────────────
def precompute_embeddings():
    """
    Read every image in photos/ folder.
    Filename stem (without extension) = gr_no.

    Example:
        photos/123014.jpg  →  gr_no = "123014"
        photos/134502.jpg  →  gr_no = "134502"
    """
    logger.info(f"Scanning '{Config.DIR_PHOTOS}/' for enrollment photos …")

    store   = {}
    enr_img = {}
    base    = Path(Config.DIR_PHOTOS)
    total   = 0
    failed  = 0

    photo_files = [
        p for p in sorted(base.glob("*"))
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    logger.info(f"Found {len(photo_files)} image file(s) in photos/")

    for img_path in photo_files:
        gr_no = img_path.stem.strip()

        if not gr_no:
            logger.warning(f"Skipping file with empty name: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            failed += 1
            continue

        emb = extract_embedding(img)
        if emb is None:
            logger.warning(f"No face detected in {img_path.name} | gr_no='{gr_no}'")
            failed += 1
            continue

        store[gr_no]   = emb.reshape(1, -1)   # (1, 512)
        enr_img[gr_no] = bgr_to_b64(img)
        logger.info(f"  ✅ Loaded: {img_path.name}  →  gr_no='{gr_no}'")
        total += 1

    with embedding_lock:
        embedding_store.clear()
        embedding_store.update(store)
        enrollment_imgs.clear()
        enrollment_imgs.update(enr_img)

    logger.info(
        f"Embeddings ready | "
        f"Loaded: {total} | "
        f"Failed: {failed} | "
        f"GR numbers: {sorted(store.keys())}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED MATCHING
# ─────────────────────────────────────────────────────────────────────────────
def match_embedding(live_emb: np.ndarray):
    """
    Compare live embedding against all stored embeddings.
    Returns: (matched_gr | None, best_score, second_score)
    """
    with embedding_lock:
        if not embedding_store:
            logger.warning("No embeddings loaded! Add photos to photos/ folder.")
            return None, 0.0, 0.0

        scores = {
            gr: float((mat @ live_emb).max())
            for gr, mat in embedding_store.items()
        }

    ranked       = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_gr, best = ranked[0]
    second        = ranked[1][1] if len(ranked) > 1 else 0.0
    margin        = best - second

    logger.info(
        f"Match result | best_gr='{best_gr}' "
        f"score={best:.3f} margin={margin:.3f} "
        f"threshold={Config.CONFIDENCE_THRESHOLD} "
        f"margin_req={Config.MARGIN_THRESHOLD}"
    )

    if best >= Config.CONFIDENCE_THRESHOLD and margin >= Config.MARGIN_THRESHOLD:
        return best_gr, best, second

    return None, best, second


# ─────────────────────────────────────────────────────────────────────────────
# FEE STATUS CHECK
# ─────────────────────────────────────────────────────────────────────────────
def is_fee_paid(fee_status: str) -> bool:
    """
    fee_status in DB is 'Paid' or 'Unpaid' (capital first letter).
    Check case-insensitively.
    """
    return str(fee_status).strip().lower() == "paid"


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CAPTURE
# ─────────────────────────────────────────────────────────────────────────────
def save_capture(bgr: np.ndarray, folder: str, label: str = "unknown") -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(folder, f"{label}_{ts}.jpg")
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def send_notification(gr_no, reason: str):
    logger.warning(f"NOTIFICATION → gr={gr_no} | reason={reason}")
    # TODO: Add SMS / email / webhook here


# ─────────────────────────────────────────────────────────────────────────────
# RESULT PUSH WITH PRIORITY HOLD
# ─────────────────────────────────────────────────────────────────────────────
def _push(result: dict):
    """
    Update latest_result with priority-hold logic.
    A result is held for Config.RESULT_HOLD_SECS seconds.
    During hold, only a higher-priority result can overwrite it.
    """
    global _last_push_time

    new_priority = _PRIORITY.get(result.get("status", "idle"), 0)

    with result_lock:
        cur_priority = _PRIORITY.get(latest_result.get("status", "idle"), 0)
        elapsed      = time.time() - _last_push_time
        still_held   = elapsed < Config.RESULT_HOLD_SECS

        if still_held and new_priority <= cur_priority:
            return   # Don't overwrite — hold current result

        latest_result.update(result)
        _last_push_time = time.time()


def _ms(t0: float) -> str:
    return f"{(time.time()-t0)*1000:.1f}ms"


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return send_from_directory("templates", "index.html")


# ── Live MJPEG stream ────────────────────────────────────────────────────────
def _mjpeg_gen():
    while True:
        with latest_frame_lock:
            frame = latest_frame
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buf.tobytes() + b"\r\n"
            )
        time.sleep(0.1)


@app.route("/live_feed")
def live_feed():
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Main upload endpoint (called by Raspberry Pi) ────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    t0 = time.time()

    # ── 1. Decode image ───────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image field"}), 400

    raw   = request.files["image"].read()
    nparr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "Cannot decode image"}), 400

    # Update live MJPEG feed
    with latest_frame_lock:
        global latest_frame
        latest_frame = frame.copy()

    cap_b64 = bgr_to_b64(frame)

    # ── 2. Extract embedding ──────────────────────────────────────────────────
    live_emb = extract_embedding(frame)
    if live_emb is None:
        save_capture(frame, Config.DIR_INVALID, "no_face")
        send_notification(None, "no_face_detected")
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
        logger.info(f"No face detected | {_ms(t0)}")
        return jsonify(result)

    # ── 3. Match against enrollment photos ───────────────────────────────────
    matched_gr, best, second = match_embedding(live_emb)

    if matched_gr is None:
        save_capture(frame, Config.DIR_INVALID, "unknown")
        send_notification(None, f"no_match score={best:.3f}")
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

    # ── 4. Get enrollment photo for matched student ───────────────────────────
    with embedding_lock:
        enr_b64 = enrollment_imgs.get(matched_gr, "")

    # ── 5. Fetch student from database ───────────────────────────────────────
    student = fetch_student(matched_gr)

    if student is None:
        save_capture(frame, Config.DIR_INVALID, f"nodb_{matched_gr}")
        send_notification(matched_gr, "not_in_database")
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
            "message":        f"GR '{matched_gr}' matched in photos but not found in database. Check gr_no matches exactly.",
            "timestamp":      datetime.now().isoformat(),
            "captured_b64":   cap_b64,
            "enrollment_b64": enr_b64,
        }
        _push(result)
        logger.warning(
            f"DB miss | matched_gr='{matched_gr}' | score={best:.3f} | {_ms(t0)}"
        )
        return jsonify(result)

    # ── 6. Check fee status ───────────────────────────────────────────────────
    paid = is_fee_paid(student["fee_status"])

    if paid:
        folder = Config.DIR_WITH_BUS
        status = "valid_with_bus"
        msg    = f"Access granted – {student['student_name']} (Bus: allowed)"
    else:
        folder = Config.DIR_WITHOUT_BUS
        status = "valid_without_bus"
        send_notification(matched_gr, "fee_unpaid")
        msg    = f"Identity confirmed – {student['student_name']} (Bus: fee unpaid)"

    save_capture(frame, folder, matched_gr)

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
        f"{status} | gr={matched_gr} | "
        f"name={student['student_name']} | "
        f"fee={student['fee_status']} | "
        f"score={best:.3f} | {_ms(t0)}"
    )
    return jsonify(result)


# ── Status poll endpoint ─────────────────────────────────────────────────────
@app.route("/upload_status", methods=["GET"])
def upload_status():
    with result_lock:
        return jsonify(latest_result)


# ── Notification endpoint ────────────────────────────────────────────────────
@app.route("/notification", methods=["POST"])
def notification():
    data = request.get_json(silent=True) or {}
    send_notification(data.get("gr_no"), data.get("reason", "unknown"))
    return jsonify({"status": "sent"})


# ── Hot-reload embeddings ────────────────────────────────────────────────────
@app.route("/reload_embeddings", methods=["POST"])
def reload_embeddings():
    secret = os.environ.get("RELOAD_SECRET", "changeme")
    if request.headers.get("X-Reload-Secret", "") != secret:
        return jsonify({"error": "Forbidden"}), 403
    threading.Thread(target=precompute_embeddings, daemon=True).start()
    return jsonify({"status": "reloading"})


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    with embedding_lock:
        n    = len(embedding_store)
        grs  = sorted(embedding_store.keys())
    return jsonify({
        "status":          "ok",
        "students_loaded": n,
        "gr_numbers":      grs,
        "timestamp":       datetime.now().isoformat(),
    })


# ── Debug endpoint ───────────────────────────────────────────────────────────
@app.route("/debug", methods=["GET"])
def debug():
    """
    Shows loaded photos and DB connectivity.
    Open: http://localhost:5000/debug
    """
    # Check photos folder
    base        = Path(Config.DIR_PHOTOS)
    photo_files = [p.name for p in sorted(base.glob("*"))
                   if p.suffix.lower() in {".jpg",".jpeg",".png"}]

    with embedding_lock:
        loaded_grs = sorted(embedding_store.keys())

    # Check DB
    db_ok    = False
    db_count = 0
    db_sample = []
    try:
        conn   = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as cnt FROM students_detail")
        db_count = cursor.fetchone()["cnt"]
        cursor.execute("SELECT gr_no, student_name, fee_status FROM students_detail LIMIT 5")
        db_sample = cursor.fetchall()
        cursor.close()
        conn.close()
        db_ok = True
    except Exception as e:
        db_sample = [{"error": str(e)}]

    return jsonify({
        "photos_folder":       Config.DIR_PHOTOS,
        "photo_files_found":   photo_files,
        "embeddings_loaded":   loaded_grs,
        "db_connected":        db_ok,
        "db_total_students":   db_count,
        "db_sample_records":   db_sample,
        "config_threshold":    Config.CONFIDENCE_THRESHOLD,
        "config_margin":       Config.MARGIN_THRESHOLD,
        "tip": "Make sure each filename in photos/ (without .jpg) exactly matches gr_no in DB",
    })


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
def startup():
    global face_app
    logger.info("=" * 55)
    logger.info("  TransBuddy Server v4.0 – Marwadi University")
    logger.info("=" * 55)

    # Test DB first
    test_db_connection()

    # Load model
    face_app = init_model()

    # Precompute embeddings
    precompute_embeddings()

    logger.info("=" * 55)
    logger.info("  Server ready on http://0.0.0.0:5000")
    logger.info("  Dashboard:  http://localhost:5000")
    logger.info("  Debug:      http://localhost:5000/debug")
    logger.info("  Health:     http://localhost:5000/health")
    logger.info("=" * 55)


if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
