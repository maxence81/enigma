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
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTORCH_ALLOC_CONF=expandable_segments:True

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

# Difficulty-3 v9 profile. These are duplicated as source defaults so direct and
# validator runs execute the same algorithm.
ENV WALL_TIME=14400 \
    HQP_ENGINE=unswap \
    HQP_DTYPE=c64 \
    HQP_CUTOFF_MODE=abs \
    HQP_US_CUTOFF=0.0002 \
    HQP_US_FINAL_CUTOFF=0.00001 \
    HQP_MIN_BOND=64 \
    HQP_US_MAXBOND=512 \
    HQP_UNSWAP_INTERVAL=20 \
    HQP_MIN_U_CONSUMED=5 \
    HQP_GREEDY_ABSORB=1 \
    HQP_GREEDY_SOFT=1.0 \
    HQP_US_TNTHRESH=200000000 \
    HQP_US_HOWS=both \
    HQP_US_MAX_ITS=1 \
    HQP_SABRE_TRIALS=200 \
    HQP_WALL_BUDGET=14100 \
    HQP_EXTRACT_RESERVE=900 \
    HQP_US_MIN_WEIGHT=0.00001 \
    HQP_XCHECK=1 \
    HQP_US_SEEDS=123 \
    HQP_MAX_RANDOM_SEEDS=0 \
    HQP_SAVE_CHECKPOINT=0

WORKDIR /app
COPY enigma_challenges /app/enigma_challenges/
COPY *.py /app/

USER miner
ENTRYPOINT ["python", "/app/hardening_quantum_proof.py"]
