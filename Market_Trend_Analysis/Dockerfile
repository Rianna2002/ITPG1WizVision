# Use Python base image
FROM python:3.11  

# Set the working directory inside the container
WORKDIR /app  

# Copy application files into the container
COPY . /app  

# Install system dependencies
RUN apt-get update && apt-get install -y sqlite3  

# Install required libraries
RUN pip install --no-cache-dir -r requirements.txt  

# Install MLflow and AWS SDK
RUN pip install mlflow boto3  

# Expose ports for both Streamlit and MLflow
EXPOSE 5000 8501  

# Set environment variables for MLflow & AWS
ENV MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

# Run MLflow server and Streamlit together
CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://aai-mta-mlflow-artifacts-bucket \
    --host 0.0.0.0 --port 5000 & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
