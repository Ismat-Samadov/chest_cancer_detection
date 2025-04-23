FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy model and application code
COPY models/chest_ct_binary_classifier_densenet_20250423_054624.keras models/
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# Set environment variables
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run with explicit host and port binding
CMD uvicorn app:app --host 0.0.0.0 --port 8080