FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p static

# Copy application code and model
COPY app.py .
COPY templates/ templates/
COPY models/ models/

# Expose the port - match with the value in render.yaml
EXPOSE 10000

# Set environment variable
ENV PORT=10000

# Command to run the API
CMD uvicorn app:app --host 0.0.0.0 --port 10000