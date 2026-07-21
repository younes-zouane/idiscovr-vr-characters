FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /app

# ── Python dependencies (cached as a separate layer) ──
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ── Torch, matching the CUDA 13.2 base image ──
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130

# ── faster-whisper pulls in CPU-only onnxruntime as a transitive dependency,
#    which conflicts with onnxruntime-gpu (see KNOWN_ISSUES.md). Force the
#    GPU version to win by reinstalling it last. ──
RUN python -m pip uninstall -y onnxruntime && \
    python -m pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.27.0

# ── Application code ──
COPY app.py .
COPY src/ ./src/
COPY wav2lip-onnx-256/ ./wav2lip-onnx-256/
COPY character_images/ ./character_images/
COPY voices/ ./voices/

EXPOSE 7860

CMD ["python", "app.py"]