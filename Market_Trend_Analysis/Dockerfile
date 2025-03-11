# Use the official Python image as base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit app port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]