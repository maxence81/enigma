FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    NUMBA_CACHE_DIR=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp

# python:3.12-slim does not set these (unlike the nvidia/cuda base). The validator
# runs bare `--gpus all`, so it relies on the image to request the CUDA "compute"
# capability. Without this the container toolkit may inject only "utility"
# (nvidia-smi, no CUDA) -> torch silently falls back to CPU. Request it explicitly.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cu130 torch

RUN pip install \
    "numpy==2.4.4" \
    "qiskit==2.4.0" \
    "qiskit-quimb==0.0.10" \
    "quimb==1.13.0" \
    "cotengra==0.7.5" \
    "autoray==0.8.10" \
    "scipy==1.17.1" \
    "qiskit-qasm3-import"

RUN useradd -r -s /sbin/nologin -d /app -U miner

ENV WALL_TIME=14400

WORKDIR /app
COPY enigma_challenges /app/enigma_challenges/
COPY *.py /app/

USER miner
ENTRYPOINT ["python", "/app/hardening_quantum_proof.py"]
