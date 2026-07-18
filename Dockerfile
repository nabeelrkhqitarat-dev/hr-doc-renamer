FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT itself; 7860 is the Hugging Face Spaces default.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
