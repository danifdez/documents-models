FROM python:3.11-slim

WORKDIR /app

RUN adduser --disabled-password --gecos '' appuser

RUN apt-get update && apt-get install -y build-essential cmake ninja-build python3-dev git && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY . .

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "python jobs.py"]