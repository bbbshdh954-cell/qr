FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

RUN mkdir -p /app/data/qr && chmod -R 777 /app/data

EXPOSE 7860

CMD ["python", "bot.py"]
