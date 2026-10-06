FROM python:3.13-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app ./app

# Copy source code
COPY src ./src

# Copy trained model and metadata
COPY artifacts ./artifacts

# Application port
EXPOSE 8000

# Start FastAPI
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "app.main:app"]