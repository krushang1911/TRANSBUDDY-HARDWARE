#!/usr/bin/env python3
"""
TransBuddy — Laptop Camera Client  v12.1.0
Marwadi University

CHANGES FROM YOUR WORKING v12:
  1. Mock GPS built-in (no external relay needed)
     GPS_MODE = "mock"  -> simulates driving through your stops
     GPS_MODE = "relay" -> uses GPS relay server (original behavior)
     GPS_MODE = "manual"-> you type lat/lon in terminal (testing)
  2. STREAM_JPEG_Q fixed (was typo 6360, now 60)
  3. Gunicorn-compatible (server side change — client unchanged)

ALL OTHER LOGIC IDENTICAL TO YOUR WORKING v12:
  • Camera open/warmup/loop unchanged
  • Offline queue unchanged
  • send_to_server unchanged
  • push_stream_frame unchanged
  • State machine (arrive/depart/cooldown) unchanged
  • Status table unchanged
"""

import sys, os, time, math, logging, threading, json, tempfile, random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("laptop_client.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("transbuddy_laptop")


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    BUS_ID = 115

    SERVER_URL  = "https://shivam2307-transbuddy-hardware.hf.space"  # no trailing slash
    TIMEOUT_SEC = 30

    # ── GPS MODE ──────────────────────────────────────────────────────────
    # "mock"   = simulates driving through stops fetched from server (no hardware)
    # "relay"  = fetches from gps_relay_server.py (original behavior)
    # "manual" = enter lat/lon in terminal for testing
    GPS_MODE      = "mock"
    GPS_RELAY_URL = "http://127.0.0.1:6000"   # used only when GPS_MODE="relay"
    GPS_POLL_SECS = 2.0
    GPS_MAX_AGE   = 30

    # ── Mock GPS settings ─────────────────────────────────────────────────
    # How fast the mock bus "drives" between stops (seconds per step)
    MOCK_TRAVEL_SECS   = 3.0   # seconds between position updates while travelling
    MOCK_DWELL_SECS    = 120   # seconds bus "stays" at each stop (must be > CAPTURE_DELAY_SECS)
    MOCK_APPROACH_STEPS = 8    # GPS steps taken while approaching stop

    # ── Stop detection ─────────────────────────────────────────────────────
    ARRIVE_RADIUS_M = 80
    DEPART_RADIUS_M = 120

    # ── Capture timing ─────────────────────────────────────────────────────
    CAPTURE_DELAY_SECS = 20
    SAMPLE_COUNT       = 4
    SAMPLE_INTERVAL    = 0.2
    COOLDOWN_SECS      = 21600

    # ── Camera ─────────────────────────────────────────────────────────────
    CAMERA_INDEX   = 0
    CAMERA_W       = 1280
    CAMERA_H       = 720
    CAMERA_WARMUP  = 5
    JPEG_QUALITY   = 88
    SHOW_PREVIEW   = False

    # ── Streaming ──────────────────────────────────────────────────────────
    STREAMING_ENABLED = True
    STREAM_INTERVAL   = 1
    STREAM_JPEG_Q     = 60        # fixed from typo 6360

    # ── Offline queue ──────────────────────────────────────────────────────
    OFFLINE_QUEUE_DIR   = "offline_queue"
    RETRY_INTERVAL_SECS = 30
    MAX_QUEUE_SIZE      = 200

    # ── Image quality ──────────────────────────────────────────────────────
    MIN_BRIGHTNESS = 15
    MAX_BRIGHTNESS = 250

    # ── Status display ─────────────────────────────────────────────────────
    STATUS_EVERY = 10


# =============================================================================
# HAVERSINE
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
            dist = d; best = s
    return best, dist


# =============================================================================
# GPS READER — supports mock, relay, manual modes
# =============================================================================
class GPSReader:
    def __init__(self):
        self._lat     = None
        self._lon     = None
        self._speed   = 0.0
        self._age     = 999
        self._lock    = threading.Lock()
        self._running = False

    @property
    def position(self):
        with self._lock: return self._lat, self._lon

    @property
    def has_fix(self):
        with self._lock:
            return self._lat is not None and self._age < Config.GPS_MAX_AGE

    def _set(self, lat, lon, speed=0.0):
        with self._lock:
            self._lat   = lat
            self._lon   = lon
            self._speed = speed
            self._age   = 0

    def start(self, stops=None):
        self._running = True
        mode = Config.GPS_MODE.lower()
        if mode == "mock":
            threading.Thread(target=self._mock_loop, args=(stops or [],),
                             daemon=True, name="gps-mock").start()
        elif mode == "relay":
            threading.Thread(target=self._relay_loop, daemon=True, name="gps-relay").start()
        elif mode == "manual":
            threading.Thread(target=self._manual_loop, daemon=True, name="gps-manual").start()
        else:
            log.error(f"Unknown GPS_MODE: {mode}")

    def _mock_loop(self, stops):
        """
        Simulates bus driving through stops in sequence.
        For each stop:
          1. Approaches from a point 200m away in MOCK_APPROACH_STEPS steps
          2. Arrives within ARRIVE_RADIUS_M
          3. Dwells for MOCK_DWELL_SECS (long enough for capture)
          4. Departs beyond DEPART_RADIUS_M
          5. Moves to next stop
        """
        if not stops:
            log.warning("Mock GPS: no stops loaded — GPS will stay at None")
            return

        log.info(f"Mock GPS started | {len(stops)} stops | "
                 f"dwell={Config.MOCK_DWELL_SECS}s per stop")

        # Start far from any stop
        start_lat = stops[0]["lat"] + 0.01
        start_lon = stops[0]["lon"] + 0.01
        self._set(start_lat, start_lon, 30.0)
        time.sleep(2)

        stop_idx = 0
        while self._running:
            stop = stops[stop_idx % len(stops)]
            stop_lat = stop["lat"]
            stop_lon = stop["lon"]

            # ── Approach: interpolate from current pos to stop ────────────
            cur_lat, cur_lon = self.position
            if cur_lat is None:
                cur_lat = stop_lat + 0.005
                cur_lon = stop_lon + 0.005

            log.info(f"Mock GPS | approaching '{stop['name']}' "
                     f"({stop_idx % len(stops) + 1}/{len(stops)})")

            steps = Config.MOCK_APPROACH_STEPS
            for i in range(steps):
                if not self._running: return
                t = (i + 1) / steps
                # Stop 40m short of centre so arrival is clean
                arrive_lat = stop_lat + (1 - t) * (cur_lat - stop_lat) * 0.05
                arrive_lon = stop_lon + (1 - t) * (cur_lon - stop_lon) * 0.05
                self._set(arrive_lat, arrive_lon, 20.0 * (1 - t))
                time.sleep(Config.MOCK_TRAVEL_SECS)

            # ── Arrive exactly at stop ────────────────────────────────────
            self._set(stop_lat, stop_lon, 0.0)
            log.info(f"Mock GPS | ARRIVED at '{stop['name']}' | "
                     f"dwelling {Config.MOCK_DWELL_SECS}s")

            # ── Dwell ─────────────────────────────────────────────────────
            dwell_end = time.time() + Config.MOCK_DWELL_SECS
            while time.time() < dwell_end and self._running:
                # Tiny jitter to simulate real GPS
                jlat = stop_lat + random.uniform(-0.00002, 0.00002)
                jlon = stop_lon + random.uniform(-0.00002, 0.00002)
                self._set(jlat, jlon, 0.0)
                time.sleep(2)

            # ── Depart — move 200m away ───────────────────────────────────
            offset = 0.002  # ~200m
            depart_lat = stop_lat + offset
            depart_lon = stop_lon + offset
            self._set(depart_lat, depart_lon, 25.0)
            log.info(f"Mock GPS | DEPARTED '{stop['name']}'")
            time.sleep(Config.MOCK_TRAVEL_SECS * 2)

            stop_idx += 1

    def _relay_loop(self):
        """Fetch GPS from relay server (original behavior)."""
        url  = f"{Config.GPS_RELAY_URL.rstrip('/')}/gps/{Config.BUS_ID}"
        tick = 0
        while self._running:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    d = r.json()
                    self._set(float(d["lat"]), float(d["lon"]),
                              float(d.get("speed", 0)))
                    if tick % 15 == 0:
                        lat, lon = self.position
                        log.info(f"GPS relay | lat={lat:.6f} lon={lon:.6f}")
                elif r.status_code == 404:
                    with self._lock: self._age = 999
                    if tick % 20 == 0:
                        log.warning("GPS relay: no fix yet")
            except requests.exceptions.ConnectionError:
                if tick % 20 == 0:
                    log.warning(f"GPS relay not reachable at {Config.GPS_RELAY_URL}")
                with self._lock: self._age = 999
            except Exception as e:
                log.debug(f"GPS relay: {e}")
                with self._lock: self._age = 999
            tick += 1
            time.sleep(Config.GPS_POLL_SECS)

    def _manual_loop(self):
        """Interactive: user types lat,lon in terminal for testing."""
        log.info("Manual GPS mode — type 'lat,lon' and press Enter (e.g. 22.303456,70.802345)")
        log.info("Type 'q' to quit")
        while self._running:
            try:
                inp = input("GPS> ").strip()
                if inp.lower() == 'q':
                    os._exit(0)
                parts = inp.split(",")
                if len(parts) == 2:
                    self._set(float(parts[0].strip()), float(parts[1].strip()), 0.0)
                    log.info(f"GPS set | lat={self._lat:.6f} lon={self._lon:.6f}")
                else:
                    log.warning("Format: lat,lon (e.g. 22.303456,70.802345)")
            except (ValueError, EOFError):
                time.sleep(1)

    def stop(self):
        self._running = False


# =============================================================================
# CAMERA MANAGER  (unchanged from your v12)
# =============================================================================
class CameraManager:
    def __init__(self):
        self._cap     = None
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = False
        self._ready   = False

    def start(self) -> bool:
        if self._running: return True
        if not self._open(): return False
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="camera").start()
        return True

    def _open(self) -> bool:
        try:
            if self._cap: self._cap.release()
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(Config.CAMERA_INDEX, backend)
            if not cap.isOpened():
                log.error(f"Camera {Config.CAMERA_INDEX} failed. Try CAMERA_INDEX 0,1,2...")
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_H)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            log.info(f"Camera {Config.CAMERA_INDEX} opened — warming up ({Config.CAMERA_WARMUP} frames)...")
            for _ in range(Config.CAMERA_WARMUP):
                cap.read(); time.sleep(0.05)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(f"Camera ready | {w}x{h}")
            self._cap = cap; self._ready = True
            return True
        except Exception as e:
            log.error(f"Camera open error: {e}"); return False

    def _loop(self):
        fail = 0
        while self._running:
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self._lock: self._frame = frame
                    fail = 0
                else:
                    fail += 1
                    if fail > 30:
                        log.warning("Camera: many failures — reconnecting...")
                        self._ready = False
                        time.sleep(2)
                        if self._open(): fail = 0
                if Config.SHOW_PREVIEW and self._frame is not None:
                    preview = cv2.resize(self._frame, (640, 360))
                    cv2.putText(preview, f"Bus {Config.BUS_ID} | TransBuddy | {Config.GPS_MODE.upper()} GPS",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 1)
                    cv2.imshow(f"TransBuddy Bus {Config.BUS_ID}", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        log.info("Q pressed — shutting down")
                        os._exit(0)
            except Exception as e:
                log.debug(f"Camera loop: {e}"); time.sleep(0.1)
            time.sleep(0.05)

    @property
    def latest_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_ready(self):
        return self._ready and self._frame is not None

    def grab_samples(self, count: int, interval: float):
        frames = []
        for i in range(count):
            f = self.latest_frame
            if f is not None: frames.append(f)
            if i < count - 1: time.sleep(interval)
        return frames

    def stop(self):
        self._running = False
        time.sleep(0.3)
        if self._cap: self._cap.release()
        if Config.SHOW_PREVIEW: cv2.destroyAllWindows()
        log.info("Camera closed")


# =============================================================================
# IMAGE QUALITY  (unchanged)
# =============================================================================
def score_image(bgr) -> float:
    if bgr is None or bgr.size == 0: return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    br   = float(np.mean(gray))
    if br < Config.MIN_BRIGHTNESS: return 0.0
    if br > Config.MAX_BRIGHTNESS: return 0.01
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    b_score   = max(0.05, 1.0 - abs(br - 128.0) / 128.0)
    s_score   = min(sharpness / 800.0, 1.0)
    return round(s_score * b_score, 4)


def select_best(frames):
    if not frames: return None, 0, 0.0
    scored = [(score_image(f), i, f) for i, f in enumerate(frames)]
    scored.sort(key=lambda x: x[0], reverse=True)
    sc, idx, fr = scored[0]
    return fr, idx, sc


# =============================================================================
# CONNECTIVITY  (unchanged)
# =============================================================================
def is_online() -> bool:
    try:
        r = requests.get(Config.SERVER_URL.rstrip("/") + "/health", timeout=4)
        return r.status_code == 200
    except: return False


# =============================================================================
# OFFLINE QUEUE  (unchanged from your v12)
# =============================================================================
class OfflineQueue:
    def __init__(self):
        self._dir = Path(Config.OFFLINE_QUEUE_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _count(self):
        return len(list(self._dir.glob("*.jpg")))

    def enqueue(self, frame, stop, gps_lat, gps_lon, quality, validated_grs=None):
        if self._count() >= Config.MAX_QUEUE_SIZE:
            log.warning("Offline queue full — dropping frame"); return False
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path  = self._dir / f"{ts}.jpg"
        meta_path = self._dir / f"{ts}.json"
        try:
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
            meta_path.write_text(json.dumps({
                "stop": stop, "gps_lat": gps_lat, "gps_lon": gps_lon,
                "quality": quality, "captured_at": datetime.now().isoformat(),
                "bus_id": Config.BUS_ID,
                "skip_gr_list": list(validated_grs) if validated_grs else [],
            }))
            log.info(f"Offline queue | saved {ts} | total={self._count()}")
            return True
        except Exception as e:
            log.error(f"Offline queue enqueue: {e}"); return False

    def process_all(self, upload_fn) -> int:
        items = sorted(self._dir.glob("*.jpg"))
        if not items: return 0
        log.info(f"Offline queue | replaying {len(items)} queued capture(s)...")
        sent = 0
        for img_path in items:
            meta_path = img_path.with_suffix(".json")
            try:
                meta  = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                frame = cv2.imread(str(img_path))
                if frame is None:
                    img_path.unlink(missing_ok=True); meta_path.unlink(missing_ok=True); continue
                result = upload_fn(
                    frame=frame, gps_lat=meta.get("gps_lat", 0),
                    gps_lon=meta.get("gps_lon", 0),
                    stop=meta.get("stop", {"name":"unknown","lat":0,"lon":0,
                                           "city":"","state":"","country":"",
                                           "display_name":"","pickup_id":""}),
                    image_quality=meta.get("quality", 0),
                    skip_gr_set=set(meta.get("skip_gr_list", [])),
                    is_offline_replay=True, captured_at=meta.get("captured_at"),
                )
                if result:
                    img_path.unlink(missing_ok=True); meta_path.unlink(missing_ok=True)
                    sent += 1; log.info(f"Offline replay OK | {img_path.name}")
                else: break
            except Exception as e:
                log.error(f"Offline replay error: {e}"); break
        return sent

    def size(self): return self._count()


# =============================================================================
# SERVER COMMUNICATION  (unchanged from your v12)
# =============================================================================
def register_bus() -> dict:
    url = Config.SERVER_URL.rstrip("/") + "/register_bus"
    try:
        r = requests.post(url, json={"bus_id": Config.BUS_ID}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            log.info(f"Bus registered | bus_no={d.get('bus_no','?')} "
                     f"type={d.get('bus_type','?')} "
                     f"driver={d.get('driver_driver_name', d.get('driver_name','?'))}")
            return d
        log.warning(f"Register bus: HTTP {r.status_code}")
    except Exception as e:
        log.warning(f"Bus registration failed: {e}")
    return {}


def load_stops() -> list:
    url = Config.SERVER_URL.rstrip("/") + "/pickup_points"
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            stops = r.json().get("stops", [])
            log.info(f"Loaded {len(stops)} stops from server")
            return stops
        except Exception as e:
            log.warning(f"Load stops attempt {attempt+1}/5: {e}")
            time.sleep(3)
    return []


def fetch_validated_today() -> set:
    url = Config.SERVER_URL.rstrip("/") + "/validated_today"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        gr_set = set(d.get("gr_list", []))
        log.info(f"Validated today ({d.get('slot','?')} slot): {len(gr_set)} GRs")
        return gr_set
    except Exception as e:
        log.warning(f"Could not fetch validated_today: {e}")
        return set()


def send_to_server(frame, gps_lat, gps_lon, stop, image_quality,
                   skip_gr_set=None, is_offline_replay=False,
                   captured_at=None) -> dict:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
    url   = Config.SERVER_URL.rstrip("/") + "/upload"
    files = {"image": ("capture.jpg", buf.tobytes(), "image/jpeg")}
    data  = {
        "bus_id":        str(Config.BUS_ID),
        "gps_lat":       str(gps_lat),
        "gps_lon":       str(gps_lon),
        "stop_name":     stop.get("name", ""),
        "stop_lat":      str(stop.get("lat", "")),
        "stop_lon":      str(stop.get("lon", "")),
        "stop_city":     stop.get("city", ""),
        "stop_state":    stop.get("state", ""),
        "stop_country":  stop.get("country", ""),
        "stop_display":  stop.get("display_name", ""),
        "pickup_id":     str(stop.get("pickup_id", "")),
        "image_quality": str(image_quality),
        "captured_at":   captured_at or datetime.now().isoformat(),
        "skip_gr_list":  ",".join(skip_gr_set) if skip_gr_set else "",
        "offline_replay":"1" if is_offline_replay else "0",
    }
    try:
        resp = requests.post(url, files=files, data=data, timeout=Config.TIMEOUT_SEC)
        resp.raise_for_status()
        result = resp.json()
        fc = result.get("face_count", 0)
        sm = result.get("summary", {})
        log.info(f"Server OK | bus={Config.BUS_ID} faces={fc} "
                 f"granted={sm.get('valid_with_bus',0)} unpaid={sm.get('unpaid',0)} "
                 f"invalid={sm.get('invalid',0)} not_uni={sm.get('not_uni',0)}")
        return result
    except requests.exceptions.ConnectionError: log.error(f"Cannot reach server: {url}")
    except requests.exceptions.Timeout:         log.error(f"Server timeout after {Config.TIMEOUT_SEC}s")
    except Exception as e:                       log.error(f"Send failed: {e}")
    return None


def push_stream_frame(cam: CameraManager) -> bool:
    frame = cam.latest_frame
    if frame is None: return False
    try:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, Config.STREAM_JPEG_Q])
        resp = requests.post(
            Config.SERVER_URL.rstrip("/") + "/stream_frame",
            files={"image": ("stream.jpg", buf.tobytes(), "image/jpeg")},
            data={"bus_id": str(Config.BUS_ID)}, timeout=5,
        )
        return resp.status_code == 200
    except: return False


# =============================================================================
# CAPTURE + DETECT  (unchanged)
# =============================================================================
def do_capture(cam, gps, stop, offline_queue, skip_gr_set=None):
    cap_lat, cap_lon = gps.position
    if cap_lat is None: log.warning("No GPS at capture time")

    log.info(f"CAPTURING {Config.SAMPLE_COUNT} samples at '{stop['name']}'")
    frames = cam.grab_samples(Config.SAMPLE_COUNT, Config.SAMPLE_INTERVAL)
    if not frames:
        log.error("No frames captured — is camera connected?"); return None

    best_frame, best_idx, best_score = select_best(frames)
    h, w = best_frame.shape[:2]
    log.info(f"Best: index={best_idx} score={best_score:.4f} | {w}x{h}")

    tmp_path = os.path.join(tempfile.gettempdir(),
                            f"transbuddy_{Config.BUS_ID}_{int(time.time())}.jpg")
    try:
        cv2.imwrite(tmp_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
        log.debug(f"Temp saved: {tmp_path}")
    except Exception as e:
        log.warning(f"Temp write failed: {e}"); tmp_path = None

    lat = cap_lat or 0.0; lon = cap_lon or 0.0
    result = send_to_server(
        frame=best_frame, gps_lat=lat, gps_lon=lon,
        stop=stop, image_quality=best_score, skip_gr_set=skip_gr_set or set(),
    )
    if tmp_path:
        try: os.remove(tmp_path)
        except: pass
    if result is None:
        log.warning("Server unreachable — saving to offline queue")
        offline_queue.enqueue(best_frame, stop, lat, lon, best_score, skip_gr_set)
    return result


# =============================================================================
# STREAMING THREAD  (unchanged)
# =============================================================================
def stream_thread(cam: CameraManager, stop_event: threading.Event):
    log.info("Stream thread started")
    fail_streak = 0
    while not stop_event.is_set():
        if cam.is_ready:
            ok = push_stream_frame(cam)
            if ok: fail_streak = 0
            else:
                fail_streak += 1
                if fail_streak > 10:
                    stop_event.wait(timeout=10); fail_streak = 0; continue
        stop_event.wait(timeout=Config.STREAM_INTERVAL)
    log.info("Stream thread stopped")


# =============================================================================
# OFFLINE RETRY THREAD  (unchanged)
# =============================================================================
def offline_retry_thread(offline_queue: OfflineQueue, stop_event: threading.Event):
    log.info("Offline retry thread started")
    while not stop_event.is_set():
        if offline_queue.size() > 0:
            if is_online():
                sent = offline_queue.process_all(send_to_server)
                if sent: log.info(f"Offline retry | replayed {sent} capture(s)")
        stop_event.wait(timeout=Config.RETRY_INTERVAL_SECS)
    log.info("Offline retry thread stopped")


# =============================================================================
# STATUS TABLE  (unchanged)
# =============================================================================
def print_status(gps, stop, dist, at_stop, elapsed_secs,
                 capture_fired, validated_count, offline_q_size, session):
    lat, lon = gps.position
    with gps._lock: age = gps._age
    lines = [
        "",
        "╔══════════════════════════════════════════════════════╗",
        f"║  TransBuddy Laptop — Bus ID: {Config.BUS_ID:<21} ║",
        f"║  GPS Mode : {Config.GPS_MODE.upper():<40} ║",
        "╠══════════════════════════════════════════════════════╣",
        f"║  GPS      : {'%.6f, %.6f' % (lat,lon) if lat else 'No fix':>36} ║",
        f"║  GPS Age  : {str(int(age))+'s':>35} ║",
        f"║  Nearest  : {(stop['name'] if stop else '—'):<40} ║",
        f"║  Distance : {('%.0fm' % dist if dist < 1e6 else '—'):<40} ║",
        f"║  At Stop  : {('YES — '+stop['name'] if at_stop else 'No'):<40} ║",
        f"║  Elapsed  : {('%ds / %ds' % (elapsed_secs, Config.CAPTURE_DELAY_SECS) if at_stop else '—'):<40} ║",
        f"║  Captured : {('YES' if capture_fired else 'No'):<40} ║",
        "╠══════════════════════════════════════════════════════╣",
        f"║  Validated this shift : {str(validated_count):<28} ║",
        f"║  Offline queue        : {str(offline_q_size):<28} ║",
        "╠══════════════════════════════════════════════════════╣",
        f"║  Granted : {str(session.get('valid',0)):<41} ║",
        f"║  Unpaid  : {str(session.get('unpaid',0)):<41} ║",
        f"║  Invalid : {str(session.get('invalid',0)):<41} ║",
        f"║  Unknown : {str(session.get('notuni',0)):<41} ║",
        "╚══════════════════════════════════════════════════════╝",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Ctrl+C to stop",
        "",
    ]
    print("\n".join(lines))


# =============================================================================
# MAIN  (unchanged from your v12 except GPS startup)
# =============================================================================
def main():
    log.info("=" * 62)
    log.info("  TransBuddy Laptop Client  v12.1.0 — Marwadi University")
    log.info(f"  Bus ID         : {Config.BUS_ID}")
    log.info(f"  Server         : {Config.SERVER_URL}")
    log.info(f"  GPS mode       : {Config.GPS_MODE}")
    log.info(f"  Camera index   : {Config.CAMERA_INDEX}")
    log.info(f"  Preview window : {'ON (Q to quit)' if Config.SHOW_PREVIEW else 'OFF'}")
    log.info(f"  Streaming      : {'ON' if Config.STREAMING_ENABLED else 'OFF'}")
    log.info(f"  Capture delay  : {Config.CAPTURE_DELAY_SECS}s")
    log.info(f"  Offline queue  : {Config.OFFLINE_QUEUE_DIR}/")
    log.info("=" * 62)

    log.info("Fetching bus stops from server...")
    stops = load_stops()
    if not stops:
        log.error("No stops loaded. Is the server running? Check SERVER_URL."); sys.exit(1)
    log.info(f"Loaded {len(stops)} stops.")

    register_bus()

    # Start GPS (passes stops so mock can drive through them)
    gps = GPSReader()
    gps.start(stops=stops)

    if Config.GPS_MODE == "mock":
        log.info("Mock GPS started — simulating bus driving through stops automatically")
        # Give mock GPS a moment to initialize
        time.sleep(2)
    elif Config.GPS_MODE == "relay":
        log.info(f"Waiting for GPS fix from relay at {Config.GPS_RELAY_URL}...")
        for i in range(60):
            if gps.has_fix:
                lat, lon = gps.position
                log.info(f"GPS fix | lat={lat:.6f} lon={lon:.6f}"); break
            time.sleep(2)
            if i % 10 == 9: log.info(f"  Still waiting... ({(i+1)*2}s)")
        else:
            log.warning("No GPS fix after 120s — continuing without")
    elif Config.GPS_MODE == "manual":
        log.info("Manual GPS mode — enter coordinates in the GPS> prompt")

    cam = CameraManager()
    log.info(f"Opening camera {Config.CAMERA_INDEX}...")
    if not cam.start():
        log.error(f"Could not open camera {Config.CAMERA_INDEX}. "
                  f"Try CAMERA_INDEX 0,1,2... or check no other app is using it.")
        sys.exit(1)
    time.sleep(0.5)

    stop_event    = threading.Event()
    offline_queue = OfflineQueue()

    if Config.STREAMING_ENABLED:
        threading.Thread(target=stream_thread, args=(cam, stop_event),
                         daemon=True, name="stream").start()

    threading.Thread(target=offline_retry_thread, args=(offline_queue, stop_event),
                     daemon=True, name="offline-retry").start()

    validated_this_shift: set = fetch_validated_today()
    session = {"valid": 0, "unpaid": 0, "invalid": 0, "notuni": 0}

    at_stop:       dict  = None
    arrived_at:    float = 0.0
    capture_fired: bool  = False
    last_trigger:  dict  = {}
    gps_tick = 0

    log.info("Tracking started. Waiting for bus to approach a stop...")
    if Config.SHOW_PREVIEW: log.info("Preview window open. Press Q in window to quit.")

    try:
        while True:
            time.sleep(1.0)
            gps_tick += 1

            lat, lon = gps.position
            if lat is None or lon is None:
                if gps_tick % 15 == 0: log.info("No GPS fix yet — waiting...")
                continue

            stop, dist = nearest_stop(lat, lon, stops)
            if stop is None: continue

            elapsed = (time.time() - arrived_at) if at_stop else 0
            if gps_tick % Config.STATUS_EVERY == 0:
                print_status(gps, stop, dist, at_stop, int(elapsed),
                             capture_fired, len(validated_this_shift),
                             offline_queue.size(), session)

            # ── STATE MACHINE ──────────────────────────────────────────────
            if at_stop is None:
                if dist <= Config.ARRIVE_RADIUS_M:
                    now  = time.time()
                    last = last_trigger.get(stop["name"], 0.0)
                    if now - last < Config.COOLDOWN_SECS:
                        rem = int(Config.COOLDOWN_SECS - (now - last))
                        log.debug(f"'{stop['name']}' cooldown — {rem//3600}h{(rem%3600)//60}m")
                        continue
                    at_stop       = stop
                    arrived_at    = time.time()
                    capture_fired = False
                    log.info(f"ARRIVED | bus={Config.BUS_ID} stop='{stop['name']}' "
                             f"dist={dist:.0f}m | detection in {Config.CAPTURE_DELAY_SECS}s")
            else:
                if dist > Config.DEPART_RADIUS_M:
                    log.info(f"DEPARTED '{at_stop['name']}' | dist={dist:.0f}m")
                    at_stop = None; capture_fired = False; continue

                rem = Config.CAPTURE_DELAY_SECS - (time.time() - arrived_at)
                if rem > 0:
                    if gps_tick % 10 == 0:
                        log.info(f"  At '{at_stop['name']}' — detection in {rem:.0f}s")
                    continue

                if capture_fired: continue

                # ── CAPTURE ────────────────────────────────────────────────
                capture_fired = True
                cur_stop      = at_stop
                result = do_capture(cam, gps, cur_stop, offline_queue, validated_this_shift)
                last_trigger[cur_stop["name"]] = time.time()

                if result:
                    fc = result.get("face_count", 0)
                    sm = result.get("summary", {})
                    session["valid"]   += sm.get("valid_with_bus", 0)
                    session["unpaid"]  += sm.get("unpaid", 0)
                    session["invalid"] += sm.get("invalid", 0)
                    session["notuni"]  += sm.get("not_uni", 0)
                    log.info(f"DONE | bus={Config.BUS_ID} stop='{cur_stop['name']}' "
                             f"faces={fc} granted={sm.get('valid_with_bus',0)} "
                             f"unpaid={sm.get('unpaid',0)}")
                    for r in result.get("results", []):
                        gr = r.get("gr_no")
                        if gr and r.get("status") not in ("not_uni", "on_cooldown"):
                            validated_this_shift.add(str(gr))
                    log.info(f"Validated this shift: {len(validated_this_shift)} students")
                else:
                    log.warning("Detection failed — saved to offline queue")

    except KeyboardInterrupt:
        log.info("Ctrl+C — shutting down...")
    finally:
        stop_event.set(); cam.stop(); gps.stop()
        log.info("Stopped.")
        log.info(f"Session — Granted:{session['valid']} Unpaid:{session['unpaid']} "
                 f"Invalid:{session['invalid']} Unknown:{session['notuni']}")


if __name__ == "__main__":
    main()