FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2

# Create a non-root user
RUN addgroup --system --gid 1001 mcpgroup && \
    adduser --system --uid 1001 --gid 1001 mcpuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
FROM base AS python-deps

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only dependency definition files
COPY pyproject.toml poetry.lock* ./

# Install runtime dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-dev

# Runtime stage
FROM base AS runtime

# Copy Python dependencies
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /config /data /plugins && \
    chown -R mcpuser:mcpgroup /app /config /data /plugins

# Switch to non-root user
USER mcpuser

# Expose port
EXPOSE 8787

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8787/health || exit 1

# Set entrypoint
ENTRYPOINT ["python", "main.py"]
