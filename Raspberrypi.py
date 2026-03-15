#!/usr/bin/env python3
"""
TransBuddy â€” Raspberry Pi GPS + Camera Client  v11.0.0
Marwadi University

BEHAVIOUR:
  1. On boot: load stops from server DB via /pickup_points
  2. GPS tracks bus position continuously (gpsd)
  3. When bus arrives within ARRIVE_RADIUS_M (80m) of a stop:
     -> Wait CAPTURE_DELAY_SECS (90s) for students to board
     -> Open camera (on-demand â€” NOT open all the time)
     -> Take SAMPLE_COUNT (4) images, spacing them SAMPLE_INTERVAL (0.3s) apart
     -> Score each image (sharpness x brightness)
     -> Select the BEST single image
     -> Send best image + GPS coordinates to server
     -> Server detects ALL faces in that image, compares each against photos/
     -> Server routes each face to correct route (A/B/C/D/E)
     -> Camera CLOSES immediately after sending
  4. Stop is on 6-hour cooldown (Pi-side) after each capture
  5. Server enforces daily slot limit (morning + evening = 2 per day per student)
"""

import sys
import os
import time
import math
import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests


# â”€â”€ LOGGING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pi_client.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("transbuddy_pi")


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # â”€â”€ Server â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    SERVER_URL  = "https://dissuasive-osseous-ethelyn.ngrok-free.dev"
    UPLOAD_PATH = "/upload"
    STOPS_PATH           = "/pickup_points"
    VALIDATED_TODAY_PATH = "/validated_today"   # GRs already validated this slot
    TIMEOUT_SEC = 30

    # â”€â”€ GPS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    GPS_MODE        = "gpsd"        # "gpsd" | "serial" | "mock"
    GPS_SERIAL_PORT = "/dev/serial0"
    GPS_SERIAL_BAUD = 9600
    GPS_POLL_SECS   = 1.0

    # â”€â”€ Stop detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ARRIVE_RADIUS_M = 80    # metres â€” triggers arrived event
    DEPART_RADIUS_M = 120   # metres â€” hysteresis for departed

    # â”€â”€ Capture timing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    CAPTURE_DELAY_SECS = 90   # wait after arriving before capture (seconds)
    SAMPLE_COUNT       = 4    # 4 samples (was 7) â€” saves 900ms with 0.3s interval
    SAMPLE_INTERVAL    = 0.3  # 0.3s between samples (was 0.6s) â€” saves 900ms total
    COOLDOWN_SECS      = 21600  # 6 hours â€” same stop won't trigger again

    # â”€â”€ Camera â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Camera is opened ON-DEMAND only (not always-on)
    CAMERA_INDEX  = 1          # confirmed working â€” change if needed
    CAMERA_WARMUP = 8          # 8 frames Ã— 50ms = 400ms warmup (was 30 = 1,500ms)
    JPEG_QUALITY  = 88

    # â”€â”€ Image quality thresholds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    MIN_BRIGHTNESS = 15        # below this = black frame
    MAX_BRIGHTNESS = 250       # above this = overexposed

    # â”€â”€ Local save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    SAVE_CAPTURES    = False  # Pi stores NOTHING permanently â€” zero SD card use
    CAPTURE_DIR      = "captures"
    SAVE_ALL_SAMPLES = False  # Never save samples on Pi â€” memory constrained

    # â”€â”€ Debug â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    LOG_GPS_EVERY = 10   # log GPS every N ticks


# =============================================================================
# HAVERSINE DISTANCE
# =============================================================================
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p)
         * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def nearest_stop(lat, lon, stops):
    if not stops:
        return None, float("inf")
    best, dist = None, float("inf")
    for s in stops:
        d = haversine(lat, lon, s["lat"], s["lon"])
        if d < dist:
            dist = d
            best = s
    return best, dist


# =============================================================================
# LOAD STOPS FROM SERVER DB
# =============================================================================
def load_stops_from_server() -> list:
    url = Config.SERVER_URL.rstrip("/") + Config.STOPS_PATH
    log.info(f"Fetching stops from: {url}")
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            stops = r.json().get("stops", [])
            log.info(f"Loaded {len(stops)} stops from server DB")
            return stops
        except Exception as e:
            log.warning(f"Attempt {attempt+1}/5 failed: {e}")
            time.sleep(3)
    log.error("Could not load stops â€” check server is running")
    return []


# =============================================================================
# FETCH ALREADY-VALIDATED GRs FROM SERVER (for this shift/slot)
# =============================================================================
def fetch_validated_today() -> set:
    """
    Asks server which GR numbers are already validated in current slot today.
    Pi uses this set to skip faces that were already processed this shift,
    avoiding unnecessary re-uploads at every bus stop.
    Returns a set of GR number strings.
    """
    url = Config.SERVER_URL.rstrip("/") + Config.VALIDATED_TODAY_PATH
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        gr_set = set(data.get("gr_list", []))
        log.info(f"Validated today ({data.get('slot','?')} slot): "
                 f"{len(gr_set)} GRs already processed")
        return gr_set
    except Exception as e:
        log.warning(f"Could not fetch validated_today: {e} â€” starting fresh")
        return set()


# =============================================================================
# GPS READER
# =============================================================================
class GPSReader:
    def __init__(self):
        self._lat = None
        self._lon = None
        self._spd = None
        self._lock = threading.Lock()
        self._gpsd_ok = False

    @property
    def position(self):
        with self._lock:
            return self._lat, self._lon

    @property
    def has_fix(self):
        with self._lock:
            return self._lat is not None and self._lon is not None

    def _update(self, lat, lon, spd=None):
        with self._lock:
            self._lat = lat
            self._lon = lon
            self._spd = spd

    def start_gpsd(self):
        def _loop():
            import gpsd
            while True:
                try:
                    if not self._gpsd_ok:
                        gpsd.connect()
                        self._gpsd_ok = True
                        log.info("GPS | gpsd connected OK")
                    p = gpsd.get_current()
                    if p.mode >= 2:
                        spd = None
                        try:
                            spd = float(p.speed()) * 3.6
                        except:
                            pass
                        self._update(float(p.lat), float(p.lon), spd)
                    else:
                        log.debug(f"GPS | no fix (mode={p.mode})")
                except Exception as e:
                    log.warning(f"GPS | error: {e}")
                    self._gpsd_ok = False
                    time.sleep(3)
                    try:
                        gpsd.connect()
                        self._gpsd_ok = True
                    except:
                        pass
                time.sleep(Config.GPS_POLL_SECS)
        threading.Thread(target=_loop, daemon=True, name="gps-gpsd").start()

    def start_serial(self):
        def _parse(line):
            try:
                p = line.strip().split(",")
                if p[0] in ("$GPRMC", "$GNRMC") and p[2] == "A":
                    la = float(p[3]); lo = float(p[5])
                    lat = int(la / 100) + (la % 100) / 60
                    lon = int(lo / 100) + (lo % 100) / 60
                    if p[4] == "S": lat = -lat
                    if p[6] == "W": lon = -lon
                    spd = None
                    try: spd = float(p[7]) * 1.852
                    except: pass
                    return lat, lon, spd
            except:
                pass
            return None, None, None

        def _loop():
            try:
                import serial
                ser = serial.Serial(Config.GPS_SERIAL_PORT, Config.GPS_SERIAL_BAUD, timeout=2)
                log.info(f"GPS | serial {Config.GPS_SERIAL_PORT} opened")
            except Exception as e:
                log.error(f"GPS | serial failed: {e}")
                return
            while True:
                try:
                    line = ser.readline().decode("ascii", errors="ignore")
                    lat, lon, spd = _parse(line)
                    if lat is not None:
                        self._update(lat, lon, spd)
                except Exception as e:
                    log.warning(f"GPS | serial error: {e}")
                    time.sleep(1)
        threading.Thread(target=_loop, daemon=True, name="gps-serial").start()

    def start_mock(self, stops):
        def _loop():
            log.info("GPS MOCK | simulating route...")
            for stop in (stops or [])[:20]:
                slat = stop["lat"] + 0.001
                slon = stop["lon"] + 0.001
                for step in range(15):
                    t = step / 15.0
                    self._update(slat + (stop["lat"] - slat) * t,
                                 slon + (stop["lon"] - slon) * t, 20.0)
                    time.sleep(0.4)
                self._update(stop["lat"], stop["lon"], 0.0)
                log.info(f"GPS MOCK | at stop: {stop['name']}")
                time.sleep(Config.CAPTURE_DELAY_SECS + 20)
                self._update(stop["lat"] + 0.002, stop["lon"] + 0.002, 20.0)
                time.sleep(3)
        threading.Thread(target=_loop, daemon=True, name="gps-mock").start()

    def start(self, stops=None):
        m = Config.GPS_MODE.lower()
        if   m == "gpsd":   self.start_gpsd()
        elif m == "serial":  self.start_serial()
        elif m == "mock":    self.start_mock(stops or [])
        else: log.error(f"Unknown GPS_MODE: {m}")


# =============================================================================
# CAMERA â€” opens ON-DEMAND, closes after every capture
# =============================================================================
class Camera:
    def __init__(self):
        self._cap   = None
        self._lock  = threading.Lock()
        self._ready = False

    def open(self) -> bool:
        """Open camera with warmup. Returns True if successful."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._ready = False

            cap = cv2.VideoCapture(Config.CAMERA_INDEX)
            if not cap.isOpened():
                log.error(f"Camera | index={Config.CAMERA_INDEX} failed to open")
                cap.release()
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            log.info(f"Camera | warming up ({Config.CAMERA_WARMUP} frames)...")
            good = False
            for i in range(Config.CAMERA_WARMUP):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    br = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                    if i % 5 == 0:
                        log.info(f"  warmup {i+1}/{Config.CAMERA_WARMUP} brightness={br:.1f}")
                    if br > Config.MIN_BRIGHTNESS:
                        good = True
                time.sleep(0.05)

            if not good:
                log.warning("Camera | warmup returned only black frames")
                # Don't abort â€” send anyway and let server decide
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(f"Camera | ready {w}x{h}")
            self._cap = cap
            self._ready = True
            return True

    def grab(self):
        """Grab the freshest frame (flushes buffer)."""
        with self._lock:
            if not self._ready or not self._cap:
                return None
            frame = None
            for _ in range(5):
                ret, f = self._cap.read()
                if ret and f is not None and f.size > 0:
                    frame = f
            return frame

    def close(self):
        """Release camera."""
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
                self._ready = False
                log.info("Camera | closed (power save)")


# =============================================================================
# IMAGE QUALITY SCORER
# =============================================================================
def score_image(bgr) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    br = float(np.mean(gray))
    if br < Config.MIN_BRIGHTNESS:
        return 0.0
    if br > Config.MAX_BRIGHTNESS:
        return 0.01
    sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()
    b_score    = max(0.05, 1.0 - abs(br - 128.0) / 128.0)
    s_score    = min(sharpness / 800.0, 1.0)
    return round(s_score * b_score, 4)


def select_best(frames):
    """Returns (best_frame, best_index, best_score)."""
    if not frames:
        return None, 0, 0.0
    scored = [(score_image(f), i, f) for i, f in enumerate(frames)]
    scored.sort(key=lambda x: x[0], reverse=True)
    sc, idx, fr = scored[0]
    return fr, idx, sc


# =============================================================================
# LOCAL SAVE
# =============================================================================
def save_local(frame, name, lat, lon, suffix=""):
    if not Config.SAVE_CAPTURES or frame is None:
        return
    Path(Config.CAPTURE_DIR).mkdir(parents=True, exist_ok=True)
    safe  = name.replace(" ", "_").replace("/", "-")[:40]
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{Config.CAPTURE_DIR}/{ts}_{safe}_{lat:.5f}_{lon:.5f}{suffix}.jpg"
    cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
    log.info(f"Saved: {fname}")
    return fname


# =============================================================================
# SEND TO SERVER
# =============================================================================
def send_to_server(frame, gps_lat, gps_lon, stop, image_quality, skip_gr_set=None):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
    url   = Config.SERVER_URL.rstrip("/") + Config.UPLOAD_PATH
    files = {"image": ("capture.jpg", buf.tobytes(), "image/jpeg")}
    data  = {
        "gps_lat":       str(gps_lat),
        "gps_lon":       str(gps_lon),
        "stop_name":     stop["name"],
        "stop_lat":      str(stop["lat"]),
        "stop_lon":      str(stop["lon"]),
        "stop_city":     stop.get("city", ""),
        "stop_state":    stop.get("state", ""),
        "stop_country":  stop.get("country", ""),
        "stop_display":  stop.get("display_name", ""),
        "pickup_id":     str(stop.get("pickup_id", "")),
        "image_quality": str(image_quality),
        "captured_at":   datetime.now().isoformat(),
        "skip_gr_list":  ",".join(skip_gr_set) if skip_gr_set else "",
    }
    log.info(f"Sending | stop={stop['name']} quality={image_quality} "
             f"gps=({gps_lat:.6f},{gps_lon:.6f})")
    try:
        resp = requests.post(url, files=files, data=data, timeout=Config.TIMEOUT_SEC)
        resp.raise_for_status()
        result = resp.json()
        fc = result.get("face_count", 0)
        sm = result.get("summary", {})
        log.info(f"Server OK | faces={fc} granted={sm.get('valid_with_bus',0)} "
                 f"unpaid={sm.get('unpaid',0)} invalid={sm.get('invalid',0)} "
                 f"not_uni={sm.get('not_uni',0)}")
        return result
    except requests.exceptions.ConnectionError:
        log.error(f"Connection refused: {url}")
    except requests.exceptions.Timeout:
        log.error(f"Timeout after {Config.TIMEOUT_SEC}s")
    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP error: {e}")
    except Exception as e:
        log.error(f"Send failed: {e}")
    return None


# =============================================================================
# CAPTURE SEQUENCE
# Opens camera -> samples -> picks best -> sends -> closes camera
# =============================================================================
def do_capture(gps, stop, skip_gr_set: set = None) -> dict:
    """
    Opens camera, takes SAMPLE_COUNT images, picks best, sends to server.
    skip_gr_set: set of GR numbers already validated this shift â€” passed
                 in upload metadata so server can do early skip checks.
    Returns server response dict or None.
    """
    cap_lat, cap_lon = gps.position
    if cap_lat is None:
        log.error("No GPS position at capture time â€” skipping")
        return None

    # â”€â”€ Open camera on-demand â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cam = Camera()
    log.info(f"Camera | opening for capture at '{stop['name']}'")
    if not cam.open():
        log.error("Camera | failed to open â€” skipping capture")
        return None

    log.info(f"CAPTURING {Config.SAMPLE_COUNT} samples at '{stop['name']}'")
    frames = []

    try:
        for i in range(Config.SAMPLE_COUNT):
            frame = cam.grab()
            if frame is None:
                log.warning(f"  Sample {i+1} â€” None from camera")
                time.sleep(Config.SAMPLE_INTERVAL)
                continue
            sc = score_image(frame)
            h, w = frame.shape[:2]
            br = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            log.info(f"  Sample {i+1}/{Config.SAMPLE_COUNT} | "
                     f"score={sc:.4f} brightness={br:.1f} size={w}x{h}")
            frames.append(frame)
            if Config.SAVE_ALL_SAMPLES:
                save_local(frame, stop["name"], cap_lat, cap_lon,
                           suffix=f"_s{i+1}_sc{sc:.3f}")
            if i < Config.SAMPLE_COUNT - 1:
                time.sleep(Config.SAMPLE_INTERVAL)
    finally:
        # â”€â”€ Close camera immediately after sampling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        cam.close()

    if not frames:
        log.error("No frames captured â€” camera problem?")
        return None

    # â”€â”€ Select best frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best_frame, best_idx, best_score = select_best(frames)
    log.info(f"Best: index={best_idx} score={best_score:.4f} "
             f"(from {len(frames)} samples)")

    if best_score == 0.0:
        log.warning("All frames score=0 (dark/black) â€” sending anyway")

    # â”€â”€ Write best frame to /tmp, send, delete immediately â”€â”€â”€â”€â”€â”€
    # /tmp is RAM-based on Pi â€” survives until reboot, no SD wear
    tmp_path = f"/tmp/transbuddy_{int(time.time())}.jpg"
    try:
        cv2.imwrite(tmp_path, best_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
        log.info(f"Temp frame written: {tmp_path}")
    except Exception as e:
        log.warning(f"Could not write temp file: {e}")
        tmp_path = None

    # â”€â”€ Send to server â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    result = send_to_server(
        frame=best_frame,
        gps_lat=cap_lat, gps_lon=cap_lon,
        stop=stop, image_quality=best_score,
        skip_gr_set=skip_gr_set or set(),
    )

    # â”€â”€ Delete temp file immediately after sending â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if tmp_path:
        try:
            os.remove(tmp_path)
            log.info(f"Temp file deleted: {tmp_path}")
        except Exception as e:
            log.warning(f"Could not delete temp file: {e}")

    return result


# =============================================================================
# MAIN LOOP
# =============================================================================
def main():
    log.info("=" * 60)
    log.info("  TransBuddy GPS Camera Client  v11.0.0")
    log.info("  Marwadi University")
    log.info("=" * 60)
    log.info(f"  Server         : {Config.SERVER_URL}")
    log.info(f"  GPS mode       : {Config.GPS_MODE}")
    log.info(f"  Camera index   : {Config.CAMERA_INDEX} (on-demand)")
    log.info(f"  Arrive radius  : {Config.ARRIVE_RADIUS_M}m")
    log.info(f"  Capture delay  : {Config.CAPTURE_DELAY_SECS}s after arriving")
    log.info(f"  Sample count   : {Config.SAMPLE_COUNT}")
    log.info(f"  Camera warmup  : {Config.CAMERA_WARMUP} frames")
    log.info(f"  Stop cooldown  : {Config.COOLDOWN_SECS//3600}h per stop")
    log.info("=" * 60)

    # â”€â”€ Load stops from server â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    stops = load_stops_from_server()
    if not stops:
        log.error("No stops loaded. Start server first. Exiting.")
        sys.exit(1)
    log.info(f"Total stops: {len(stops)}")

    # â”€â”€ Start GPS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    gps = GPSReader()
    gps.start(stops=stops)

    # â”€â”€ Wait for GPS fix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if Config.GPS_MODE != "mock":
        log.info("Waiting for GPS fix...")
        waited = 0
        while not gps.has_fix and waited < 120:
            time.sleep(2); waited += 2
            log.info(f"  ... {waited}s")
        if not gps.has_fix:
            log.warning("No GPS fix after 120s â€” continuing anyway")
        else:
            lat, lon = gps.position
            log.info(f"GPS fix | lat={lat:.6f} lon={lon:.6f}")

    # â”€â”€ Load already-validated GRs for this shift â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    validated_this_shift: set = fetch_validated_today()

    # â”€â”€ State machine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    at_stop:       dict  = None
    arrived_at:    float = 0.0
    capture_fired: bool  = False
    last_trigger:  dict  = {}   # stop_name -> last trigger timestamp
    gps_tick = 0

    log.info("GPS tracking started â€” waiting for bus to approach a stop...")

    try:
        while True:
            time.sleep(Config.GPS_POLL_SECS)
            gps_tick += 1

            lat, lon = gps.position
            if lat is None or lon is None:
                log.debug("No GPS fix yet")
                continue

            stop, dist = nearest_stop(lat, lon, stops)
            if stop is None:
                continue

            if gps_tick % Config.LOG_GPS_EVERY == 0:
                log.info(f"GPS | lat={lat:.6f} lon={lon:.6f} | "
                         f"nearest='{stop['name']}' dist={dist:.0f}m")

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # STATE MACHINE
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if at_stop is None:
                # Not at any stop currently
                if dist <= Config.ARRIVE_RADIUS_M:
                    now  = time.time()
                    last = last_trigger.get(stop["name"], 0.0)
                    if now - last < Config.COOLDOWN_SECS:
                        rem = int(Config.COOLDOWN_SECS - (now - last))
                        log.debug(f"'{stop['name']}' cooldown â€” {rem//3600}h {(rem%3600)//60}m left")
                        continue
                    # ARRIVED
                    at_stop       = stop
                    arrived_at    = time.time()
                    capture_fired = False
                    log.info(f"ARRIVED at '{stop['name']}' | dist={dist:.0f}m | "
                             f"capture in {Config.CAPTURE_DELAY_SECS}s")

            else:
                # Currently at a stop
                if dist > Config.DEPART_RADIUS_M:
                    log.info(f"DEPARTED '{at_stop['name']}' | dist={dist:.0f}m")
                    at_stop       = None
                    capture_fired = False
                    continue

                elapsed  = time.time() - arrived_at
                rem_secs = Config.CAPTURE_DELAY_SECS - elapsed

                if rem_secs > 0:
                    if gps_tick % 10 == 0:
                        log.info(f"  At '{at_stop['name']}' â€” capture in {rem_secs:.0f}s")
                    continue

                if capture_fired:
                    continue  # already captured for this stop visit

                # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                # TRIGGER CAPTURE
                # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                capture_fired = True
                current_stop  = at_stop  # snapshot before departure

                result = do_capture(gps, current_stop,
                                          skip_gr_set=validated_this_shift)

                # Record trigger time for cooldown
                last_trigger[current_stop["name"]] = time.time()

                if result:
                    fc = result.get("face_count", 0)
                    sm = result.get("summary", {})
                    log.info(
                        f"DONE | stop='{current_stop['name']}' "
                        f"faces={fc} "
                        f"granted={sm.get('valid_with_bus', 0)} "
                        f"unpaid={sm.get('unpaid', 0)} "
                        f"invalid={sm.get('invalid', 0)} "
                        f"not_uni={sm.get('not_uni', 0)}"
                    )
                    # Update local validated set with newly processed GRs
                    for r in result.get("results", []):
                        gr = r.get("gr_no")
                        if gr and r.get("status") not in ("not_uni", "on_cooldown"):
                            validated_this_shift.add(str(gr))
                    log.info(f"Validated this shift so far: {len(validated_this_shift)} students")
                else:
                    log.warning("Server send failed â€” check server logs")

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        log.info("TransBuddy Pi Client stopped.")


if __name__ == "__main__":
    main()
