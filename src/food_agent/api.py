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


class FoodQuery(BaseModel):
    food: str


def _summary(trace: dict) -> dict:
    return {"answer": trace["answer"], "tool_calls": len(trace["steps"]),
            "cost_usd": trace.get("cost_usd"), "trace_id": trace["id"]}


@app.get("/health")
def health():
    return {"ok": True, "agent": AGENT_NAME}


@app.post("/analyze")
def analyze_food(q: FoodQuery):
    return _summary(analyze(q.food, verbose=False))


@app.post("/analyze-label")
async def analyze_label(file: UploadFile = File(...)):
    suffix = Path(file.filename or "label.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    label = read_label(path)
    if not label.get("found"):
        return JSONResponse({"error": "Could not read an ingredients label from that image."}, status_code=422)
    trace = analyze(f"Analyze this product's ingredients: {label.get('ingredients_text')}", verbose=False)
    return {"label": label, **_summary(trace)}


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>{AGENT_NAME}</title><style>
body{{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;color:#1a1a1a}}
h1{{margin-bottom:2px}} .sub{{color:#666;margin-top:0}}
input,button{{font-size:15px;padding:8px 10px}} button{{cursor:pointer;background:#137a3f;color:#fff;border:0;border-radius:6px}}
#out{{white-space:pre-wrap;background:#f6f8f6;border-left:3px solid #137a3f;padding:12px;margin-top:16px;border-radius:4px}}
.row{{margin:14px 0}}</style></head><body>
<h1>{AGENT_NAME}</h1><p class=sub>{AGENT_TAGLINE}</p>
<div class=row><input id=food size=34 placeholder="e.g. McDonald's Big Mac">
<button onclick=byName()>Analyze food</button></div>
<div class=row>…or a label photo: <input type=file id=img accept=image/*>
<button onclick=byLabel()>Read label</button></div>
<div id=out>Ask about any food, or upload a label.</div>
<script>
const out=document.getElementById('out');
async function byName(){{ out.textContent='Analyzing…';
 const r=await fetch('/analyze',{{method:'POST',headers:{{'Content-Type':'application/json'}},
  body:JSON.stringify({{food:document.getElementById('food').value}})}});
 const j=await r.json(); out.textContent=j.answer+'\\n\\n['+j.tool_calls+' tool calls · $'+j.cost_usd+']'; }}
async function byLabel(){{ const f=document.getElementById('img').files[0]; if(!f)return;
 out.textContent='Reading label…'; const fd=new FormData(); fd.append('file',f);
 const r=await fetch('/analyze-label',{{method:'POST',body:fd}}); const j=await r.json();
 out.textContent=(j.answer||j.error||'')+(j.tool_calls?'\\n\\n['+j.tool_calls+' tool calls · $'+j.cost_usd+']':''); }}
</script></body></html>"""
