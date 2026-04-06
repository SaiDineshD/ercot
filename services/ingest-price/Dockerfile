FROM python:3.11-slim
WORKDIR /app
COPY services/ingest-price/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/_shared/health_http.py ./health_http.py
COPY services/ingest-price/ingest_price.py ./ingest_price.py
EXPOSE 8080
CMD ["python", "-u", "ingest_price.py"]
