import cv2
import requests
import time

SERVER_URL = "https://dissuasive-osseous-ethelyn.ngrok-free.dev/upload"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imwrite("temp.jpg", frame)

    try:
        with open("temp.jpg", "rb") as f:
            response = requests.post(
                SERVER_URL,
                files={"image": f},
                timeout=10
            )

        print("Response:", response.json())

    except Exception as e:
        print("Error:", e)

    time.sleep(2)
