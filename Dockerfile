# syntax=docker/dockerfile:1

# 1. Builder Stage: Install all dependencies, including dev tools
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install all dependencies from requirements.txt into the venv
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# 2. Final Stage: Copy only the necessary files
FROM python:3.11-slim AS final
COPY --from=builder /opt/venv /opt/venv

# Copy application code
WORKDIR /app
COPY . .

# Set path to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PORT=8080

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
