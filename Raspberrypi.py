"""
TransBuddy Bus Face Verification – Raspberry Pi Client
Marwadi University – Production System
Version: 3.0.0

Uses NGROK public URL to send frames to server.
No Flask on Pi side – pure HTTP POST client only.

Setup:
    1. Start ngrok on server PC:
           ngrok http 5000
    2. Copy the https URL (e.g. https://abc123.ngrok-free.app)
    3. Paste it below in Config.SERVER_URL
    4. Run this script on Raspberry Pi:
           python3 raspberry_pi.py

Requirements (install on Raspberry Pi):
    pip3 install requests opencv-python-headless
"""

import time
import logging
import requests
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← Edit these values before running
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # ── Paste your ngrok URL here (no trailing slash) ──────────────────────
    # Example: "https://abc123.ngrok-free.app"
    SERVER_URL = "https://dissuasive-osseous-ethelyn.ngrok-free.dev"
    
    # Upload endpoint
    UPLOAD_ENDPOINT = "/upload"

    # Camera index (0 = first camera, Pi CSI or USB webcam)
    CAMERA_INDEX = 0

    # Frame resolution
    FRAME_WIDTH  = 640
    FRAME_HEIGHT = 480

    # JPEG quality for upload (lower = smaller payload = faster upload)
    JPEG_QUALITY = 80

    # How long to wait for server response (seconds)
    # Increase if ngrok is slow on your network
    NETWORK_TIMEOUT = 10

    # Sleep between captures (seconds)
    CAPTURE_SLEEP_SEC = 1.5

    # Sleep when no face detected (seconds)
    NO_FACE_SLEEP_SEC = 0.4

    # Minimum face size as fraction of frame (reject faces too far away)
    MIN_FACE_FRAC = 0.03

    # Maximum face size as fraction of frame (reject faces too close)
    MAX_FACE_FRAC = 0.90

    # Print result details to terminal
    VERBOSE = True


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers= [logging.StreamHandler()]
)
logger = logging.getLogger("transbuddy-pi")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def validate_config():
    if "YOUR-NGROK-URL" in Config.SERVER_URL:
        logger.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.error("  ERROR: Please set your ngrok URL in Config.SERVER_URL")
        logger.error("  Example: https://abc123.ngrok-free.app")
        logger.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        raise SystemExit(1)

    if not Config.SERVER_URL.startswith("https://"):
        logger.warning("SERVER_URL does not start with https:// — are you sure this is correct?")

    logger.info(f"Server URL : {Config.SERVER_URL}")
    logger.info(f"Upload URL : {Config.SERVER_URL}{Config.UPLOAD_ENDPOINT}")


# ─────────────────────────────────────────────────────────────────────────────
# HAAR CASCADE (lightweight on-device face gate)
# Only decides WHETHER to send frame — actual recognition is server-side ArcFace
# ─────────────────────────────────────────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────
def init_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Cannot open camera. Check connection and CAMERA_INDEX.")
        raise RuntimeError("Camera not available")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)    # Prevent stale frames
    logger.info(f"Camera ready ({Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT})")
    return cap


# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTION GATE
# ─────────────────────────────────────────────────────────────────────────────
def has_valid_face(frame: np.ndarray) -> bool:
    """
    Returns True if a face of acceptable size is visible.
    Used only as a capture gate — NOT for identity verification.
    Identity is handled by server-side ArcFace.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)          # Improve low-light detection
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (55, 55),
        flags        = cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return False

    frame_area = Config.FRAME_WIDTH * Config.FRAME_HEIGHT
    for (x, y, w, h) in faces:
        frac = (w * h) / frame_area
        if Config.MIN_FACE_FRAC <= frac <= Config.MAX_FACE_FRAC:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE FRAME TO JPEG (in memory, no disk write)
# ─────────────────────────────────────────────────────────────────────────────
def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# SEND FRAME TO SERVER VIA NGROK
# ─────────────────────────────────────────────────────────────────────────────
def send_frame(jpeg_bytes: bytes) -> dict | None:
    """
    POST JPEG frame to ngrok server /upload endpoint.

    ngrok requires the 'ngrok-skip-browser-warning' header
    to bypass the ngrok browser warning page.
    """
    url = Config.SERVER_URL.rstrip("/") + Config.UPLOAD_ENDPOINT

    try:
        response = requests.post(
            url,
            files   = {"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            headers = {
                # Required for ngrok free tier to skip interstitial page
                "ngrok-skip-browser-warning": "true",
                "User-Agent": "TransBuddy-Pi/3.0"
            },
            timeout = Config.NETWORK_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout after {Config.NETWORK_TIMEOUT}s — server slow or ngrok issue")
    except requests.exceptions.ConnectionError:
        logger.warning("Connection error — check ngrok URL or internet connection")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP {e.response.status_code} from server: {e}")
    except requests.exceptions.JSONDecodeError:
        logger.warning("Server returned non-JSON response — possible ngrok warning page")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# HANDLE RESULT
# ─────────────────────────────────────────────────────────────────────────────
def handle_result(result: dict):
    """
    Process server response and trigger appropriate action.
    Extend this to drive GPIO pins (LED / buzzer / relay / LCD).
    """
    status = result.get("status", "unknown")

    if status == "valid_with_bus":
        logger.info(
            f"✅ ACCESS GRANTED  | "
            f"{result.get('name')} | "
            f"GR: {result.get('gr_no')} | "
            f"Route: {result.get('route')} | "
            f"Conf: {result.get('confidence', 0):.3f}"
        )
        gpio_signal("green")

    elif status == "valid_without_bus":
        logger.info(
            f"⚠️  FEE UNPAID      | "
            f"{result.get('name')} | "
            f"GR: {result.get('gr_no')} | "
            f"Fee: {result.get('fee_status')}"
        )
        gpio_signal("yellow")

    elif status == "invalid_person":
        logger.warning(
            f"❌ UNKNOWN PERSON  | "
            f"Score: {result.get('confidence', 0):.3f} | "
            f"{result.get('message', '')}"
        )
        gpio_signal("red")

    elif status == "invalid_database":
        logger.warning(
            f"🔴 NOT IN DATABASE | "
            f"GR: {result.get('gr_no')} | "
            f"{result.get('message', '')}"
        )
        gpio_signal("red")

    else:
        logger.warning(f"Unknown status from server: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# GPIO SIGNAL (extend for your hardware)
# ─────────────────────────────────────────────────────────────────────────────
def gpio_signal(color: str):
    """
    Drive GPIO pins based on result.

    Uncomment and adapt for your Raspberry Pi wiring:

    import RPi.GPIO as GPIO
    GREEN_PIN  = 17
    YELLOW_PIN = 27
    RED_PIN    = 22
    BUZZ_PIN   = 23

    GPIO.setmode(GPIO.BCM)
    GPIO.setup([GREEN_PIN, YELLOW_PIN, RED_PIN, BUZZ_PIN], GPIO.OUT, initial=GPIO.LOW)

    if color == "green":
        GPIO.output(GREEN_PIN, GPIO.HIGH)
        time.sleep(1.5)
        GPIO.output(GREEN_PIN, GPIO.LOW)

    elif color == "yellow":
        for _ in range(3):                   # Blink yellow 3 times
            GPIO.output(YELLOW_PIN, GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(YELLOW_PIN, GPIO.LOW)
            time.sleep(0.3)

    elif color == "red":
        GPIO.output(RED_PIN, GPIO.HIGH)
        GPIO.output(BUZZ_PIN, GPIO.HIGH)
        time.sleep(1.0)
        GPIO.output(RED_PIN, GPIO.LOW)
        GPIO.output(BUZZ_PIN, GPIO.LOW)
    """
    pass   # Remove this line once GPIO is wired


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION TEST
# ─────────────────────────────────────────────────────────────────────────────
def test_connection() -> bool:
    """
    Test if server is reachable via ngrok before starting main loop.
    Hits the /health endpoint.
    """
    url = Config.SERVER_URL.rstrip("/") + "/health"
    logger.info(f"Testing connection to {url} …")
    try:
        r = requests.get(
            url,
            headers = {"ngrok-skip-browser-warning": "true"},
            timeout = 8
        )
        r.raise_for_status()
        data = r.json()
        logger.info(
            f"Server reachable ✅ | "
            f"Students loaded: {data.get('students', '?')} | "
            f"Status: {data.get('status')}"
        )
        return True
    except Exception as e:
        logger.error(f"Cannot reach server: {e}")
        logger.error("Check: 1) ngrok is running  2) SERVER_URL is correct  3) Internet OK")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  TransBuddy Pi Client v3.0 – Marwadi University")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    validate_config()

    # Test server connectivity before starting camera
    if not test_connection():
        logger.error("Aborting — fix server connection first.")
        return

    cap = init_camera()
    no_face_count = 0

    try:
        while True:
            # ── Read frame ───────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Frame read failed — retrying …")
                time.sleep(0.5)
                continue

            # ── Face gate ─────────────────────────────────────────────────────
            if not has_valid_face(frame):
                no_face_count += 1
                if no_face_count % 20 == 0:
                    logger.info(f"Waiting for face … ({no_face_count} frames scanned)")
                time.sleep(Config.NO_FACE_SLEEP_SEC)
                continue

            no_face_count = 0
            logger.info("Face detected — encoding and uploading …")

            # ── Encode JPEG in memory ─────────────────────────────────────────
            try:
                jpeg_bytes = encode_jpeg(frame)
            except RuntimeError as e:
                logger.error(f"Encoding error: {e}")
                time.sleep(0.5)
                continue

            logger.info(f"Sending {len(jpeg_bytes) // 1024} KB to {Config.SERVER_URL} …")

            # ── Upload via ngrok ──────────────────────────────────────────────
            result = send_frame(jpeg_bytes)

            if result:
                handle_result(result)
            else:
                logger.warning("No valid response — will retry next cycle")

            # ── Wait before next capture ──────────────────────────────────────
            time.sleep(Config.CAPTURE_SLEEP_SEC)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        cap.release()
        logger.info("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
