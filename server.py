import os
import cv2
import numpy as np
import datetime
import shutil
from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
from scipy.spatial.distance import cosine
import pymysql

# ================= CONFIG =================

PHOTO_FOLDER = "photos"
UPLOAD_FOLDER = "static"
INVALID_FOLDER = "static/invalid_captures"

SIMILARITY_THRESHOLD = 0.40

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INVALID_FOLDER, exist_ok=True)

app = Flask(__name__)

latest_result = {"status": "waiting"}

# ================= DATABASE =================

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "transbuddy_db_1",
    "cursorclass": pymysql.cursors.DictCursor
}

def get_db():
    return pymysql.connect(**DB_CONFIG)

def fetch_student(gr_no):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM students_detail WHERE gr_no=%s",
                (gr_no,)
            )
            return cur.fetchone()
    finally:
        conn.close()

# ================= LOAD MODEL =================

print("Loading FaceNet model...")
model = DeepFace.build_model("Facenet")
print("Model loaded.")

# ================= BUILD STUDENT EMBEDDINGS =================

print("Building student embeddings...")

student_embeddings = {}

for file in os.listdir(PHOTO_FOLDER):
    if file.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(PHOTO_FOLDER, file)
        try:
            embedding = DeepFace.represent(
                img_path=path,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            gr_no = os.path.splitext(file)[0]
            student_embeddings[gr_no] = embedding
        except Exception as e:
            print("Skipping:", file)

print(f"Loaded {len(student_embeddings)} students.")

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
        return jsonify({"status": "error", "error": "No image file"})

    image = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")
    image.save(save_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        live_embedding = DeepFace.represent(
            img_path=save_path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        best_match = None
        best_distance = 999

        for gr_no, stored_embedding in student_embeddings.items():
            dist = cosine(live_embedding, stored_embedding)
            if dist < best_distance:
                best_distance = dist
                best_match = gr_no

        if best_distance < SIMILARITY_THRESHOLD:

            student = fetch_student(best_match)

            if student:
                latest_result = {
                    "status": "valid",
                    "gr_no": best_match,
                    "distance": float(best_distance),
                    "details": student,
                    "timestamp": timestamp
                }
            else:
                latest_result = {
                    "status": "invalid_database",
                    "gr_no": best_match,
                    "distance": float(best_distance),
                    "timestamp": timestamp
                }

        else:
            invalid_path = os.path.join(
                INVALID_FOLDER,
                f"unknown_{int(datetime.datetime.now().timestamp())}.jpg"
            )
            shutil.copy(save_path, invalid_path)

            latest_result = {
                "status": "invalid_person",
                "distance": float(best_distance),
                "timestamp": timestamp
            }

        return jsonify(latest_result)

    except Exception as e:
        latest_result = {
            "status": "error",
            "error": str(e)
        }
        return jsonify(latest_result)

# ================= START =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
