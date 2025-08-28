FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    tesseract-ocr tesseract-ocr-spa \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OCR_ENABLED=false \
    CORS_ORIGINS="http://localhost:5173,http://localhost:3000"

EXPOSE 8000
CMD ["uvicorn", "servidor:app", "--host", "0.0.0.0", "--port", "8000"]
