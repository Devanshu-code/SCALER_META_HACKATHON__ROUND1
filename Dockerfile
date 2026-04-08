FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Run the server from the server directory
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
