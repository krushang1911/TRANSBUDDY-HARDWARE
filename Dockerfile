# =============================================================================
# TransBuddy Server — Dockerfile for Hugging Face Spaces
# Multi-Bus Face Verification System (Marwadi University)
# =============================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System libraries ──────────────────────────────────────────────────────────
# build-essential + cmake  : compile insightface C extensions
# libstdc++6               : REQUIRED at runtime by onnxruntime binary
# libgl1 + libglib2.0-0    : required by opencv-python-headless
# libgomp1                 : OpenMP threading used by onnxruntime
# curl                     : healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libstdc++6 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python packages — ORDER MATTERS ──────────────────────────────────────────
# 1. Cython           — insightface setup.py requires it before compile
# 2. numpy==1.26.4    — pinned; onnxruntime & insightface C-ABI depends on this
# 3. onnxruntime 1.19.2 — first release with NumPy 2.x ABI support
# 4. opencv           — pre-built headless wheel, no source compile needed
# 5. insightface      — compiles against numpy already installed
# 6. app deps         — flask, gunicorn, mysql, etc.
# 7. numpy re-pinned  — FINAL guard: prevents any upstream dep from upgrading numpy
RUN pip install --upgrade pip setuptools wheel "Cython>=0.29.0" \
 && pip install --prefer-binary "numpy==1.26.4" \
 && pip install --prefer-binary "onnxruntime==1.19.2" \
 && pip install --prefer-binary "opencv-python-headless==4.8.1.78" \
 && pip install "insightface==0.7.3" \
 && pip install \
        "flask==3.0.3" \
        "gunicorn==21.2.0" \
        "mysql-connector-python==8.3.0" \
        "requests==2.31.0" \
        "Pillow==10.3.0" \
        "huggingface_hub>=0.20.0" \
 && pip install --prefer-binary --force-reinstall "numpy==1.26.4"

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p \
    captures/with_bus \
    captures/without_bus \
    captures/invalid_captures \
    captures/not_uni_student \
    proof_images \
    offline_queue

# ── Application files ─────────────────────────────────────────────────────────
COPY server.py .
COPY templates/ templates/
COPY photos/ photos/

# ── Hugging Face Spaces runtime ───────────────────────────────────────────────
ENV HF_SPACES_ENV="1"

# HF Spaces REQUIRES port 7860 — do not change this
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["gunicorn", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "180", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "server:app"]