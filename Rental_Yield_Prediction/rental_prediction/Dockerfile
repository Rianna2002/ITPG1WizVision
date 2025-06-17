FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p models data
COPY models/ /app/models/

# Set environment variables
ENV PYTHONPATH=/app

# Expose port for Streamlit only
EXPOSE 8501

# Run Streamlit directly
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]