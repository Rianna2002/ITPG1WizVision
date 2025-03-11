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

# Create directories for MLflow and models
RUN mkdir -p mlruns models data

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Expose ports for Streamlit and MLflow UI
EXPOSE 8501 5000

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
mlflow ui --host 0.0.0.0 & \n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]