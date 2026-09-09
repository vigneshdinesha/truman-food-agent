"""
FastAPI service for Truman.

Endpoints:
  GET  /                 -> minimal HTML demo (type a food, or upload a label photo)
  POST /analyze          -> {"food": "..."}  analyze a named food
  POST /analyze-label    -> multipart image upload; reads the label then analyzes
  GET  /health           -> liveness

Run:  PYTHONPATH=src .venv/bin/uvicorn food_agent.api:app --reload
"""
from __future__ import annotations
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .config import AGENT_NAME, AGENT_TAGLINE
from .agent import analyze
from .vision import read_label

app = FastAPI(title=f"{AGENT_NAME} — {AGENT_TAGLINE}")

# Cost/abuse guard for the public demo: cap total analyses per day so a shared
# API key can't be run up. In-memory (resets on restart) — fine for a portfolio demo.
import datetime
_DEMO_DAILY_CAP = 300
_usage = {"day": "", "count": 0}


def _demo_ok() -> bool:
    today = datetime.date.today().isoformat()
    if _usage["day"] != today:
        _usage.update(day=today, count=0)
    _usage["count"] += 1
    return _usage["count"] <= _DEMO_DAILY_CAP


_CAP_MSG = ("This is a portfolio demo with a small daily budget, and it's been hit for today. "
            "Try again tomorrow, or run it yourself from the GitHub repo.")


class FoodQuery(BaseModel):
    food: str


def _summary(trace: dict) -> dict:
    return {"answer": trace["answer"], "verdict": trace.get("verdict"),
            "tool_calls": len(trace["steps"]), "cost_usd": trace.get("cost_usd"),
            "trace_id": trace["id"]}


@app.get("/health")
def health():
    return {"ok": True, "agent": AGENT_NAME}


@app.post("/analyze")
def analyze_food(q: FoodQuery):
    if not _demo_ok():
        return JSONResponse({"answer": _CAP_MSG}, status_code=429)
    return _summary(analyze(q.food, verbose=False))


@app.post("/analyze-label")
async def analyze_label(file: UploadFile = File(...)):
    if not _demo_ok():
        return JSONResponse({"error": _CAP_MSG}, status_code=429)
    suffix = Path(file.filename or "label.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    label = read_label(path)
    if not label.get("found"):
        return JSONResponse({"error": "Could not read an ingredients label from that image."}, status_code=422)
    trace = analyze(f"Analyze this product's ingredients: {label.get('ingredients_text')}", verbose=False)
    return {"label": label, **_summary(trace)}


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8") \
    .replace("{{AGENT_NAME}}", AGENT_NAME).replace("{{AGENT_TAGLINE}}", AGENT_TAGLINE)


@app.get("/", response_class=HTMLResponse)
def home():
    return _INDEX_HTML
