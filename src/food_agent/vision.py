"""
Vision: read an ingredients list off a photo of a food label using gpt-4o-mini
(multimodal). Returns structured ingredients the agent can then analyze — this is
the second input path (the first being a typed food name -> Open Food Facts).

Usage:
    from food_agent.vision import read_label
    read_label("/path/to/label.jpg")     # local file
    read_label("https://.../label.jpg")   # or a URL
"""
from __future__ import annotations
import base64, json, mimetypes
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from .config import CHAT_MODEL, require_openai_key

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_PROMPT = """You are reading a photo of a food's INGREDIENTS label.
Extract ONLY what is printed. Return JSON:
{"found": true/false,
 "product_name": string or null,
 "ingredients_text": the full ingredients line as printed,
 "ingredients_list": [each ingredient as a separate string]}
If the image is not a legible ingredients label, return {"found": false}."""


def _to_data_url(image: str) -> str:
    if image.startswith("http://") or image.startswith("https://"):
        return image
    p = Path(image)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def read_label(image: str) -> dict:
    """OCR + structure a food label photo into ingredients."""
    require_openai_key()
    client = OpenAI()
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": _PROMPT},
            {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
        ]}])
    out = json.loads(resp.choices[0].message.content)
    out["source"] = "vision"
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m food_agent.vision <image_path_or_url>")
        raise SystemExit(1)
    print(json.dumps(read_label(sys.argv[1]), indent=2, ensure_ascii=False))
