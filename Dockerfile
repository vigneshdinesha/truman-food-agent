FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code + curated corpus + prebuilt vector index (data/index.npz)
COPY . .

ENV PYTHONPATH=/app/src
EXPOSE 8080
# OPENAI_API_KEY is provided at runtime as a Fly secret (never baked into the image)
CMD ["sh", "-c", "uvicorn food_agent.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
