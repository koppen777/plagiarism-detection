FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

WORKDIR /app

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache --break-system-packages --no-cache-dir -r requirements.txt

COPY . .

CMD ["python","run.py"]
