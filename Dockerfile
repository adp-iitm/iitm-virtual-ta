# Use Python 3.8 as the base image
FROM python:3.8-slim

# Install system dependencies for pytesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port (Render sets $PORT)
EXPOSE 8000

# Start the app using shell form to allow $PORT substitution
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT