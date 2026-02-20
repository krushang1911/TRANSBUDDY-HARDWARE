import cv2
import requests
import time

SERVER_URL = "https://dissuasive-osseous-ethelyn.ngrok-free.dev/upload"
CAPTURE_INTERVAL = 5

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected.")
    exit()

print("Camera started...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Capture failed.")
        continue

    cv2.imwrite("capture.jpg", frame)

    try:
        with open("capture.jpg", "rb") as img:
            response = requests.post(
                SERVER_URL,
                files={"image": img},
                timeout=120
            )

        print("Status:", response.status_code)

        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print("Server Error:", response.text)

    except Exception as e:
        print("Request Error:", e)

    time.sleep(CAPTURE_INTERVAL)
