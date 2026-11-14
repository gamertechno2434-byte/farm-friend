FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY index.html farm_friend_app.py ./

EXPOSE 8000

CMD ["sh", "-c", "uvicorn farm_friend_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
