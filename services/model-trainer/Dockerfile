FROM python:3.11-slim
WORKDIR /app

COPY services/model-trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ /app/models/
COPY services/_shared/health_http.py /app/health_http.py
COPY services/model-trainer/train.py .

EXPOSE 8080
CMD ["python", "-u", "train.py"]
