# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy prediction script
COPY web_service/predict_flask.py .

# Copy local model
COPY web_service/model ./model

# Copy requirements file
COPY requirements.txt .

# Copy source code (for any custom preprocessing, utils, etc.)
COPY src ./src

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 9696

# Run the Flask app
CMD ["python", "predict_flask.py"]