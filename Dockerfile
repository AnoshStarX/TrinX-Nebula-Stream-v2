FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

# Install system dependencies for better async performance
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libdbus-1-3 libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    libwayland-client0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright chromium for crawl4ai (shared path readable by appuser)
RUN playwright install chromium && chmod -R o+rx /opt/playwright

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on (internal container port)
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default to one worker for deterministic in-process fallback behavior.
# Scale with UVICORN_WORKERS when shared persistence/cache are available.
CMD ["sh", "-c", "uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers ${UVICORN_WORKERS:-1} \
    --loop uvloop \
    --http httptools \
    --proxy-headers \
    --forwarded-allow-ips '*' \
    --timeout-keep-alive 30 \
    --limit-concurrency 200 \
    --backlog 2048 \
    --no-server-header \
    --no-access-log"]
