FROM python:3.11-slim
WORKDIR /app
COPY services/ingest-weather/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/_shared/health_http.py ./health_http.py
COPY services/ingest-weather/ingest_weather.py ./ingest_weather.py
EXPOSE 8080
CMD ["python", "-u", "ingest_weather.py"]
