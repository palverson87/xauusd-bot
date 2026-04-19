FROM python:3.11-slim

# matplotlib Agg backend needs libglib; curl for healthcheck probes
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY *.py ./

# Pre-build matplotlib font cache so first request isn't slow
RUN python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot"

# Reports land here; mount a Railway Volume at /app/reports for persistence
RUN mkdir -p reports

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Tell the app to bind on all interfaces so Railway can route to it
    HOST=0.0.0.0

EXPOSE 8050

CMD ["python", "app.py"]
