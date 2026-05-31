FROM python:3.11-slim

ENV HYPERZERO_DEVICE=cpu \
    HYPERZERO_PRELOAD_MODEL=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-render.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-render.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn services.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
