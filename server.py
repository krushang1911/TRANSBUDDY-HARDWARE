import os
import cv2
import numpy as np
import datetime
import shutil
from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import pymysql

# ================= CONFIG =================

PHOTO_FOLDER = "photos"
UPLOAD_FOLDER = "static"
WITH_BUS_FOLDER = "static/with_bus"
WITHOUT_BUS_FOLDER = "static/without_bus"
INVALID_FOLDER = "static/invalid"

MATCH_THRESHOLD = 0.62  # production tuned

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WITH_BUS_FOLDER, exist_ok=True)
os.makedirs(WITHOUT_BUS_FOLDER, exist_ok=True)
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
print("Model Loaded.")

# ================= BUILD EMBEDDINGS =================

print("Building student embeddings...")

student_embeddings = {}
counts = {}

for file in os.listdir(PHOTO_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(PHOTO_FOLDER, file)
        try:
            emb = DeepFace.represent(
                img_path=path,
                model_name="Facenet",
                enforce_detection=True
            )[0]["embedding"]

            emb = np.array(emb)
            emb = emb / np.linalg.norm(emb)

            gr = file.split("_")[0].split(".")[0]

            if gr not in student_embeddings:
                student_embeddings[gr] = emb
                counts[gr] = 1
            else:
                student_embeddings[gr] += emb
                counts[gr] += 1

        except:
            print("Skipped:", file)

# Average embeddings
for gr in student_embeddings:
    student_embeddings[gr] /= counts[gr]
    student_embeddings[gr] /= np.linalg.norm(student_embeddings[gr])

print("Students Loaded:", len(student_embeddings))

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
        return jsonify({"status": "error", "error": "No file"})

    image = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")
    image.save(save_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        emb = DeepFace.represent(
            img_path=save_path,
            model_name="Facenet",
            enforce_detection=True
        )[0]["embedding"]

        emb = np.array(emb)
        emb = emb / np.linalg.norm(emb)

        best_similarity = 0
        best_gr = None

        for gr, stored_emb in student_embeddings.items():
            similarity = np.dot(stored_emb, emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_gr = gr

        if best_similarity >= MATCH_THRESHOLD:

            student = fetch_student(best_gr)

            if student:

                fee_status = student.get("fee_status", "unpaid")

                if fee_status.lower() == "paid":
                    target_folder = WITH_BUS_FOLDER
                    final_status = "valid_with_bus"
                else:
                    target_folder = WITHOUT_BUS_FOLDER
                    final_status = "valid_without_bus"

                shutil.copy(
                    save_path,
                    os.path.join(target_folder, f"{best_gr}_{int(datetime.datetime.now().timestamp())}.jpg")
                )

                latest_result = {
                    "status": final_status,
                    "gr_no": best_gr,
                    "similarity": float(best_similarity),
                    "details": student,
                    "timestamp": timestamp
                }

            else:
                latest_result = {
                    "status": "invalid_database",
                    "similarity": float(best_similarity),
                    "timestamp": timestamp
                }

        else:
            shutil.copy(
                save_path,
                os.path.join(INVALID_FOLDER, f"unknown_{int(datetime.datetime.now().timestamp())}.jpg")
            )

            latest_result = {
                "status": "invalid_person",
                "similarity": float(best_similarity),
                "timestamp": timestamp
            }

        return jsonify(latest_result)

    except Exception as e:
        latest_result = {"status": "error", "error": str(e)}
        return jsonify(latest_result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
