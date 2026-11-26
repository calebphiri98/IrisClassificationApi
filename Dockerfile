FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY iris_model.pkl .
COPY serve_model.py .

EXPOSE 80

CMD ["python", "serve_model.py"]
