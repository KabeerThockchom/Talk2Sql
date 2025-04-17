FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
COPY .env .env
# Create necessary directories
RUN mkdir -p /app/data/databases /app/data/training_data /app/data/audio_cache

# Create symbolic links
RUN ln -sf /app/data/databases /app/databases && \
    ln -sf /app/data/training_data /app/training_data && \
    ln -sf /app/data/audio_cache /app/audio_cache

# Expose the port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"] 