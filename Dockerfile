# AI-Enhanced PDF Extraction System
# Docker Configuration für lokales Ollama + Python Setup

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create Documents directory
RUN mkdir -p Documents

# Expose port for potential web interface
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "ai_pdf_processor.py"]
