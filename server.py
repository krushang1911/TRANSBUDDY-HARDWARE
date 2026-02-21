import os
import cv2
import numpy as np
import datetime
import shutil
import requests
from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import pymysql

# ================= CONFIG =================

PHOTO_FOLDER = "photos"
UPLOAD_FOLDER = "static"
INVALID_FOLDER = "static/invalid_captures"
WITH_BUS_FOLDER = "with_bus"
WITHOUT_BUS_FOLDER = "without_bus"

SIMILARITY_THRESHOLD = 0.55  # tuned for ArcFace

NOTIFICATION_URL = "https://your-ngrok-link/notification"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INVALID_FOLDER, exist_ok=True)
os.makedirs(WITH_BUS_FOLDER, exist_ok=True)
os.makedirs(WITHOUT_BUS_FOLDER, exist_ok=True)

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

def fetch_student(gr_no):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM students_detail WHERE gr_no=%s", (gr_no,))
            return cur.fetchone()
    finally:
        conn.close()

# ================= LOAD ARC FACE MODEL =================

print("Loading ArcFace model...")
model = DeepFace.build_model("ArcFace")
print("Model Loaded.")

# ================= BUILD EMBEDDINGS =================

print("Building embedding matrix...")

gr_list = []
embedding_list = []

for file in os.listdir(PHOTO_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(PHOTO_FOLDER, file)
        try:
            emb = DeepFace.represent(
                img_path=path,
                model_name="ArcFace",
                enforce_detection=False
            )[0]["embedding"]

            emb = np.array(emb)
            emb = emb / np.linalg.norm(emb)

            gr = os.path.splitext(file)[0]
            gr_list.append(gr)
            embedding_list.append(emb)

        except:
            print("Skipping:", file)

embedding_matrix = np.array(embedding_list)

print("Students Loaded:", len(gr_list))

# ================= NOTIFICATION =================

def send_notification(route_name, gr_no, student_details, fee_status, timestamp):
    try:
        payload = {
            "route": route_name,
            "gr_no": gr_no,
            "details": student_details,
            "fee_status": fee_status,
            "timestamp": timestamp
        }
        requests.post(NOTIFICATION_URL, json=payload, timeout=3)
    except:
        pass

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
        return jsonify({"status": "error"})

    image = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")
    image.save(save_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        live_emb = DeepFace.represent(
            img_path=save_path,
            model_name="ArcFace",
            enforce_detection=False
        )[0]["embedding"]

        live_emb = np.array(live_emb)
        live_emb = live_emb / np.linalg.norm(live_emb)

        # ⚡ Vectorized similarity
        similarities = np.dot(embedding_matrix, live_emb)

        best_index = np.argmax(similarities)
        best_similarity = similarities[best_index]
        best_gr = gr_list[best_index]

        if best_similarity > SIMILARITY_THRESHOLD:

            student = fetch_student(best_gr)

            if student:
                fee_status = student.get("fee_status", "").lower()

                if fee_status == "paid":
                    target = WITH_BUS_FOLDER
                    final_status = "valid_with_bus"
                else:
                    target = WITHOUT_BUS_FOLDER
                    final_status = "valid_without_bus"

                    send_notification(
                        "fee_unpaid",
                        best_gr,
                        student,
                        fee_status,
                        timestamp
                    )

                shutil.copy(
                    save_path,
                    os.path.join(target, f"{best_gr}_{int(datetime.datetime.now().timestamp())}.jpg")
                )

                latest_result = {
                    "status": final_status,
                    "gr_no": best_gr,
                    "similarity": float(best_similarity),
                    "details": student,
                    "timestamp": timestamp
                }

            else:
                latest_result = {"status": "invalid_database"}
                send_notification("invalid_database", best_gr, None, "unknown", timestamp)

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

            send_notification("invalid_person", "unknown", None, "unknown", timestamp)

        return jsonify(latest_result)

    except Exception as e:
        latest_result = {"status": "error", "error": str(e)}
        return jsonify(latest_result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
