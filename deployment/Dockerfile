# Predictive Maintenance Platform — FastAPI serving image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for scientific Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy the application source.
COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY scripts/ ./scripts/

EXPOSE 8000

# Train a model on first boot if no artifacts are mounted, then serve.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
