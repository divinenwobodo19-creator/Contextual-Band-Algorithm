FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY linucb_brain/ ./linucb_brain/

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "linucb_brain.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
