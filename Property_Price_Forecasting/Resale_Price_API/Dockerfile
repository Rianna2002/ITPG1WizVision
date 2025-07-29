FROM public.ecr.aws/lambda/python:3.9

# Copy your code
COPY . ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the CMD to your handler function in main.py
CMD ["main.lambda_handler"]
