"""
Tool: look up a food's ingredient list from Open Food Facts (free, open DB of
millions of products). Used when the user *names* a food ("McDonald's Big Mac")
instead of photographing a label.

No API key required. Open Food Facts asks for a descriptive User-Agent, and its
search server is sometimes overloaded (503) — so we use a retry session.
"""
from __future__ import annotations
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
    _RETRY = Retry(total=4, backoff_factor=0.8,
                   status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
except Exception:  # pragma: no cover
    _RETRY = None

_UA = {"User-Agent": "food-safety-agent/0.1 (portfolio project; contact via github)"}
_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"
_PRODUCT = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
_FIELDS = "code,product_name,brands,ingredients_text,ingredients,additives_tags,allergens_tags"

_session = requests.Session()
_session.headers.update(_UA)
if _RETRY:
    _session.mount("https://", HTTPAdapter(max_retries=_RETRY))


def _clean(product: dict) -> dict:
    """Normalize an Open Food Facts product into the fields we care about."""
    additives = [a.split(":")[-1] for a in (product.get("additives_tags") or [])]  # 'en:e330' -> 'e330'
    ingredients = [i.get("text") for i in (product.get("ingredients") or []) if i.get("text")]
    return {
        "found": True,
        "source": "openfoodfacts",
        "barcode": product.get("code"),
        "product_name": product.get("product_name") or "(unnamed product)",
        "brand": product.get("brands"),
        "ingredients_text": (product.get("ingredients_text") or "").strip(),
        "ingredients_list": ingredients,
        "additives": additives,               # E-number codes detected by OFF
        "allergens": [a.split(":")[-1] for a in (product.get("allergens_tags") or [])],
    }


def lookup_by_name(query: str, timeout: int = 20) -> dict:
    """Search Open Food Facts by product name; return the best-matching product's ingredients."""
    params = {"search_terms": query, "search_simple": 1, "action": "process",
              "json": 1, "page_size": 5, "fields": _FIELDS, "sort_by": "unique_scans_n"}
    try:
        r = _session.get(_SEARCH, params=params, timeout=timeout)
        r.raise_for_status()
        products = r.json().get("products", [])
    except Exception as e:
        return {"found": False, "query": query, "error": str(e),
                "note": "Open Food Facts search was unreachable; a barcode lookup or a label photo is more reliable."}
    for p in products:
        if p.get("ingredients_text") or p.get("ingredients"):
            out = _clean(p)
            out["query"] = query
            return out
    return {"found": False, "query": query,
            "note": "No product with an ingredient list matched. Try a more specific name or a photo of the label."}


def lookup_by_barcode(barcode: str, timeout: int = 20) -> dict:
    """Exact lookup by product barcode (most reliable when available)."""
    try:
        r = _session.get(_PRODUCT.format(barcode=barcode), params={"fields": _FIELDS}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"found": False, "barcode": barcode, "error": str(e)}
    if data.get("status") == 1 and data.get("product"):
        return _clean(data["product"])
    return {"found": False, "barcode": barcode, "note": "No product found for that barcode."}


if __name__ == "__main__":  # quick manual test (no API key needed)
    import json, sys
    q = " ".join(sys.argv[1:]) or "coca cola"
    print(json.dumps(lookup_by_name(q), indent=2, ensure_ascii=False)[:2000])
