# Use official Python runtime as a parent image
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend code
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Optional: write Google credentials from env var at runtime
COPY start.sh /app/start.sh
# Normalize potential Windows CRLF line endings and ensure executable bit
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Expose port (Render/Vercel/containers will set PORT env)
EXPOSE 8000

# Default command uses PORT if provided (Render sets PORT)
ENV PORT=8000
CMD ["/app/start.sh"]
