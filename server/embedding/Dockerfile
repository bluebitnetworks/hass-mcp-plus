FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_NAME="all-MiniLM-L6-v2"

# Create a non-root user
RUN addgroup --system --gid 1001 embeddinggroup && \
    adduser --system --uid 1001 --gid 1001 embeddinguser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory with proper permissions
RUN mkdir -p /app/model_cache && \
    chown -R embeddinguser:embeddinggroup /app

# Switch to non-root user
USER embeddinguser

# Download model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/${MODEL_NAME}')"

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Command to run the service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
