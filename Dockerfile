# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Copy only pyproject.toml first to leverage Docker layer caching
COPY pyproject.toml .

# Create a dummy README.md to prevent setuptools from failing if it looks for it
RUN touch README.md

# Install dependencies
RUN python -m pip install --upgrade pip \
    && pip install build setuptools \
    && pip install .

# Now copy the rest of the application code
COPY . .

# Cloud Run provides $PORT; default to 8080 for local runs
    ENV PORT=8080 \
    USE_JSON_LOGGING=true

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
