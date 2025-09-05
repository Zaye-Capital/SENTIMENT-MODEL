# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# (Optional) tools some HF models may use; keep image small.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download all model weights/tokenizers into the image
RUN python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
models = [
 "cardiffnlp/twitter-roberta-base-sentiment-latest",
 "j-hartmann/sentiment-roberta-large-english-3-classes",
 "nlptown/bert-base-multilingual-uncased-sentiment",
]
for m in models:
    AutoTokenizer.from_pretrained(m, cache_dir="/models")
    AutoModelForSequenceClassification.from_pretrained(m, cache_dir="/models")
print("Models downloaded into /models")
PY

COPY svc.py .

EXPOSE 8000
CMD uvicorn svc:app --host 0.0.0.0 --port ${PORT:-8000}
