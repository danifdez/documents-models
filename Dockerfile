FROM python:3.11-slim

WORKDIR /app

RUN adduser --disabled-password --gecos '' appuser

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY . .

USER appuser

EXPOSE 8000

RUN python -m spacy download en_core_web_trf

CMD ["sh", "-c", "python jobs.py"]