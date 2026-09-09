"""
Generate a COMPREHENSIVE, ENRICHED additive reference corpus.

Source of truth: the Open Food Facts additives taxonomy (~680 E-number additives)
for the factual scaffold (function, EFSA-evaluation status, ANSES flags), enriched
with a plain-language Wikipedia summary per additive (resolved via the taxonomy's
wikidata links for accuracy). Only sourced, attributable text — no LLM-generated
content, so the corpus can't be poisoned with hallucinations.

Output: data/corpus/additives_reference.md  (one '## ' section per additive)
Run:  PYTHONPATH=src python -m food_agent.rag.build_additives
"""
from __future__ import annotations
import requests, json, time
from ..config import CORPUS_DIR

CACHE = CORPUS_DIR.parent / ".wiki_cache.json"   # data/.wiki_cache.json (gitignored)

TAXO = "https://static.openfoodfacts.org/data/taxonomies/additives.json"
WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "food-safety-agent/0.1 (portfolio project; food additive reference)"}
OUT = CORPUS_DIR / "additives_reference.md"


def _en(field):
    return (field or {}).get("en")


def _clean_name(entry, key):
    raw = _en(entry.get("name")) or key.replace("en:", "").upper()
    return raw.split(" - ", 1)[1].strip() if " - " in raw else raw.strip()


def _trim(extract: str) -> str:
    extract = extract.strip()
    if len(extract) > 600:
        cut = extract[:600]
        extract = cut[: cut.rfind(". ") + 1] or cut
    return extract


def fetch_summaries(titles: list[str]) -> dict[str, str]:
    """Batch Wikipedia intro extracts with a disk cache + backoff, so coverage
    ACCUMULATES across runs and survives Wikipedia rate-limiting. Only fetches
    titles not already cached; a failed (rate-limited) batch is left uncached to
    retry next run; a genuine 'no article' is cached as '' to avoid re-fetching."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    uniq = sorted(set(titles))
    todo = [t for t in uniq if t not in cache]
    print(f"  {len(uniq) - len(todo)} cached · fetching {len(todo)} new (batched, polite)...")

    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        data = None
        for attempt in range(4):
            try:
                r = requests.get(WIKI_API, headers=UA, timeout=45, params={
                    "action": "query", "format": "json", "prop": "extracts",
                    "exintro": 1, "explaintext": 1, "redirects": 1, "exlimit": 20,
                    "titles": "|".join(batch)})
                if r.status_code == 200:
                    data = r.json().get("query", {}); break
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))         # backoff on failure/throttle
        if data is None:
            continue                                 # leave batch uncached; retry next run
        norm = {n["from"]: n["to"] for n in data.get("normalized", [])}
        redir = {x["from"]: x["to"] for x in data.get("redirects", [])}
        by_title = {p["title"]: p["extract"] for p in data.get("pages", {}).values()
                    if p.get("extract") and "may refer to" not in p.get("extract", "")[:40]}
        for t in batch:
            t2 = redir.get(norm.get(t, t), norm.get(t, t))
            ex = by_title.get(t2) or by_title.get(norm.get(t, t)) or by_title.get(t)
            cache[t] = _trim(ex) if ex else ""       # cache genuine misses as ''
        CACHE.write_text(json.dumps(cache))          # persist incrementally
        time.sleep(0.4)                              # be polite to Wikipedia

    return {t: cache[t] for t in uniq if cache.get(t)}


def wikidata_titles(qids: list[str]) -> dict[str, str]:
    """Batch-resolve wikidata Q-ids -> exact English Wikipedia article titles (50/call)."""
    out = {}
    uniq = sorted(set(qids))
    for i in range(0, len(uniq), 50):
        try:
            r = requests.get("https://www.wikidata.org/w/api.php", headers=UA, timeout=45, params={
                "action": "wbgetentities", "ids": "|".join(uniq[i:i + 50]),
                "props": "sitelinks", "sitefilter": "enwiki", "format": "json"})
            for qid, e in r.json().get("entities", {}).items():
                t = e.get("sitelinks", {}).get("enwiki", {}).get("title")
                if t:
                    out[qid] = t
        except Exception:
            continue
    return out


def build() -> int:
    tax = requests.get(TAXO, headers=UA, timeout=60).json()
    entries = {k: e for k, e in tax.items() if _en(e.get("name"))}

    # exact article titles via wikidata where available; clean name otherwise
    qid_by_key = {k: _en(e.get("wikidata")) for k, e in entries.items() if _en(e.get("wikidata"))}
    wd = wikidata_titles(list(qid_by_key.values()))
    title_by_key = {k: (wd.get(qid_by_key.get(k, "")) or _clean_name(e, k))
                    for k, e in entries.items()}

    uniq = set(title_by_key.values())
    print(f"Fetching Wikipedia summaries for {len(uniq)} articles (batched)...")
    summaries = fetch_summaries(list(title_by_key.values()))
    got = len(summaries)
    print(f"  got {got}/{len(uniq)} summaries")

    lines = ["# Additive Reference (Open Food Facts taxonomy + Wikipedia)", "",
             "> Comprehensive, sourced reference for E-number additives: factual fields from",
             "> the Open Food Facts taxonomy and EFSA, plus a plain-language Wikipedia summary.",
             "> No opinions or generated content — all attributable. Deeper health context for",
             "> common additives lives in the hand-curated docs.", ""]
    n = 0
    for key, e in sorted(entries.items()):
        enum = _en(e.get("e_number"))
        title = f"E{enum} — {_clean_name(e, key)}" if enum else _clean_name(e, key)
        cls = _en(e.get("additives_classes"))
        cls = ", ".join(c.replace("en:", "").strip() for c in cls.split(",")) if cls else None
        sec = [f"## {title}"]
        if cls:
            sec.append(f"- Function: {cls}")
        summ = summaries.get(title_by_key[key])
        if summ:
            sec.append(f"- What it is: {summ} (Source: Wikipedia.)")
        elif _en(e.get("description")):
            sec.append(f"- Description: {_en(e.get('description'))}")
        if _en(e.get("anses_additives_of_interest")) == "yes":
            sec.append("- Flagged by ANSES (French food-safety agency) as an **additive of interest**.")
        efsa_date = _en(e.get("efsa_evaluation_date"))
        sec.append(f"- EFSA safety evaluation on record ({efsa_date})." if efsa_date
                   else "- No EFSA re-evaluation date on record in the taxonomy.")
        if _en(e.get("vegan")):
            sec.append(f"- Vegan: {_en(e.get('vegan'))}.")
        sec.append("- Source: Open Food Facts additives taxonomy"
                   + (", EFSA" if efsa_date else "") + (", Wikipedia." if summ else "."))
        lines.append("\n".join(sec) + "\n")
        n += 1

    OUT.write_text("\n".join(lines))
    print(f"Wrote {n} additive entries ({got} with Wikipedia context) -> {OUT}")
    return n


if __name__ == "__main__":
    build()
