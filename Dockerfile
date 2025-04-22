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

# Expose the port that the application listens on
EXPOSE 8000

# Environment variable for PORT - Render will set this
ENV PORT=8000

# Command to run the API
CMD uvicorn app:app --host 0.0.0.0 --port $PORT