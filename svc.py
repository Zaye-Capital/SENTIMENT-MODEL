# svc.py
import os
from typing import Optional, Tuple, Any, Dict
from fastapi import FastAPI, Header, HTTPException, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


# -------- Config --------
API_KEY = (os.getenv("SENTIMENT_SVC_API_KEY") or "").strip()
REQUIRE_AUTH = len(API_KEY) > 0
PORT = int(os.getenv("PORT", "8000"))
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

HF_CACHE = os.getenv("TRANSFORMERS_CACHE") or os.getenv("HF_HOME") or None

# -------- App --------
app = FastAPI(title="Sentiment Service", version="1.0.0")

# -------- Model IDs --------
SOCIAL_ID  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EN3C_ID    = "j-hartmann/sentiment-roberta-large-english-3-classes"
MULTI5S_ID = "nlptown/bert-base-multilingual-uncased-sentiment"

def make_pipe(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=os.getenv("HF_HOME"))
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=os.getenv("HF_HOME"))
    # top_k=None ≈ return_all_scores=True (list-of-lists of dicts)
    return pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None)

# Eager load (fastest). If you need lower RAM, you can lazy-load on first use.
pipe_social  = make_pipe(SOCIAL_ID)
pipe_en3c    = make_pipe(EN3C_ID)
pipe_multi5s = make_pipe(MULTI5S_ID)

# -------- Auth helpers --------
def _extract_bearer(auth_header: Optional[str]) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None

def _auth_guard(x_api_key: Optional[str], authorization: Optional[str]):
    if not REQUIRE_AUTH:
        return
    token = x_api_key or _extract_bearer(authorization)
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------- Payload parsing (supports your current HF-style payload) --------
def parse_incoming_payload(data: Dict[str, Any]) -> Tuple[str, int]:
    """
    Accepts either:
      { "text": "...", "max_length": 256 }
    OR Hugging Face style:
      { "inputs": "...", "parameters": {"max_length": 256, "truncation": true, "return_all_scores": true }, "options": {...} }
    """
    if "text" in data:
        text = str(data.get("text") or "")
        max_len = int(data.get("max_length") or 256)
        return text, max_len

    if "inputs" in data:
        text = str(data.get("inputs") or "")
        params = data.get("parameters") or {}
        max_len = int(params.get("max_length") or 256)
        return text, max_len

    raise HTTPException(status_code=400, detail="Bad payload: expected 'text' or 'inputs'.")

def _classify(pipe, text: str, max_len: int):
    # Important: pipeline returns [[{label,score}...]] when return_all_scores=True
    out = pipe(text, truncation=True, max_length=max_len)
    return out  # keep nested list to match HF Inference API

# -------- Routes --------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/social-en")
async def social_en(req: Request,
                    x_api_key: Optional[str] = Header(default=None),
                    authorization: Optional[str] = Header(default=None)):
    _auth_guard(x_api_key, authorization)
    data = await req.json()
    text, max_len = parse_incoming_payload(data)
    return _classify(pipe_social, text, max_len)

@app.post("/review/en-3c")
async def review_en3c(req: Request,
                      x_api_key: Optional[str] = Header(default=None),
                      authorization: Optional[str] = Header(default=None)):
    _auth_guard(x_api_key, authorization)
    data = await req.json()
    text, max_len = parse_incoming_payload(data)
    return _classify(pipe_en3c, text, max_len)

@app.post("/review/multi-5s")
async def review_multi5s(req: Request,
                         x_api_key: Optional[str] = Header(default=None),
                         authorization: Optional[str] = Header(default=None)):
    _auth_guard(x_api_key, authorization)
    data = await req.json()
    text, max_len = parse_incoming_payload(data)
    return _classify(pipe_multi5s, text, max_len)
