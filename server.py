"""
=============================================================
  Student Face Verification System — server.py
=============================================================
"""

import os
import datetime
import shutil
import time
import logging
import pymysql
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from deepface import DeepFace

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static")
STUDENT_PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
INVALID_LOG_DIR = os.path.join(UPLOAD_FOLDER, "invalid_captures")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STUDENT_PHOTOS_DIR, exist_ok=True)
os.makedirs(INVALID_LOG_DIR, exist_ok=True)

SIMILARITY_THRESHOLD = 0.55

latest_result = {"status": "waiting"}

# ================= DATABASE =================

DB_CONFIG: dict[str, str | type] = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "transbuddy",
    "cursorclass": pymysql.cursors.DictCursor
}

def get_db():
    return pymysql.connect(**DB_CONFIG)

def fetch_student(gr_no):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM students WHERE gr_no=%s", (gr_no,))
            return cur.fetchone()
    finally:
        conn.close()

# ================= FACE MATCH =================

def match_face(live_img_path):

    best_gr = None
    best_dist = float("inf")

    for file in os.listdir(STUDENT_PHOTOS_DIR):
        if not file.lower().endswith((".gif", ".jpg", ".jpeg", ".png")):
            continue

        student_path = os.path.join(STUDENT_PHOTOS_DIR, file)

        try:
            result = DeepFace.verify(
                img1_path=live_img_path,
                img2_path=student_path,
                model_name="Facenet",
                detector_backend="retinaface",
                enforce_detection=True,
                distance_metric="cosine"
            )

            dist = result["distance"]

            if dist < best_dist:
                best_dist = dist
                best_gr = os.path.splitext(file)[0]

        except Exception as e:
            log.warning(f"Verification error for {file}: {e}")
            continue

    if best_dist <= SIMILARITY_THRESHOLD:
        return best_gr, best_dist

    return None, best_dist

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_status")
def upload_status():
    return jsonify(latest_result)

@app.route("/upload", methods=["POST"])
def upload():
    global latest_result

    if "image" not in request.files:
        return jsonify({"status": "error"}), 400

    image = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")
    image.save(save_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gr_no, dist = match_face(save_path)

    if gr_no:
        student = fetch_student(gr_no)

        if student:
            latest_result = {
                "status": "valid",
                "gr_no": gr_no,
                "distance": round(dist, 4),
                "details": student,
                "timestamp": timestamp
            }
            return jsonify(latest_result), 200

        else:
            latest_result = {
                "status": "invalid_database",
                "gr_no": gr_no,
                "distance": round(dist, 4),
                "timestamp": timestamp
            }
            return jsonify(latest_result), 200

    else:
        latest_result = {
            "status": "invalid_person",
            "distance": round(dist, 4),
            "timestamp": timestamp
        }
        return jsonify(latest_result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
