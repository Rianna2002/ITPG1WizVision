# Use official Python image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements and app files into container
COPY requirements.txt ./
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
