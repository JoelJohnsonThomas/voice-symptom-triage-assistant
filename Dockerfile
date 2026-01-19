# Multi-stage Dockerfile for Voice Symptom Intake & Documentation Assistant

# Stage 1: Base image with Python
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: CPU-only variant (for development/testing)
FROM base as cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DEVICE=cpu
ENV ENABLE_GPU=false

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: GPU-enabled variant (for production)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as gpu

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV DEVICE=cuda
ENV ENABLE_GPU=true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
