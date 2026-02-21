import cv2
import requests
import time

SERVER_URL = "https://dissuasive-osseous-ethelyn.ngrok-free.dev/upload"

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    try:
        response = requests.post(
            SERVER_URL,
            files={"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=4
        )
        print(response.json())
    except:
        print("Server unreachable")

    time.sleep(0.1)
