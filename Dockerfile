FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Copy project files into the container
COPY . /app

# Basic system deps + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=3600
ENV PIP_NO_CACHE_DIR=1

# Install GPU-enabled PyTorch and Python deps
RUN pip3 install -r requirements_api.txt

# Expose the Flask port
EXPOSE 7860

ENV PYTHONPATH="/app"

# Start the Flask API server
CMD ["python3", "api_server.py"]
