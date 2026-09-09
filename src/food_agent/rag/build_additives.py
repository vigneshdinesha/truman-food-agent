"""
Generate a COMPREHENSIVE additive reference corpus from the Open Food Facts
additives taxonomy (~680 E-number additives). This gives Truman baseline, sourced
coverage of essentially every additive that appears on a food label anywhere —
factual, structured fields only (no LLM-generated content, so the corpus can't be
poisoned with hallucinations). The hand-curated docs remain the premium layer for
common/important additives, and PubChem is the live fallback for anything else.

Output: data/corpus/additives_reference.md  (one '## ' section per additive)
Run:  PYTHONPATH=src python -m food_agent.rag.build_additives
"""
from __future__ import annotations
import requests
from ..config import CORPUS_DIR

TAXO = "https://static.openfoodfacts.org/data/taxonomies/additives.json"
_UA = {"User-Agent": "food-safety-agent/0.1 (portfolio project)"}
OUT = CORPUS_DIR / "additives_reference.md"


def _en(field: dict | None):
    return (field or {}).get("en")


def _clean_name(entry: dict, key: str) -> str:
    raw = _en(entry.get("name")) or key.replace("en:", "").upper()
    # names look like "E330 - Citric acid" -> take the descriptive part if present
    return raw.split(" - ", 1)[1].strip() if " - " in raw else raw.strip()


def build() -> int:
    tax = requests.get(TAXO, headers=_UA, timeout=60).json()
    lines = [
        "# Additive Reference (Open Food Facts additives taxonomy)",
        "",
        "> Comprehensive, sourced baseline for E-number additives. Fields are factual",
        "> data from the Open Food Facts taxonomy and EFSA evaluations — not opinions.",
        "> For deeper health context on common additives, see the hand-curated docs.",
        "",
    ]
    n = 0
    for key, e in sorted(tax.items()):
        name_en = _en(e.get("name"))
        if not name_en:
            continue
        enum = _en(e.get("e_number"))
        title = f"E{enum} — {_clean_name(e, key)}" if enum else _clean_name(e, key)
        cls = _en(e.get("additives_classes"))
        cls = ", ".join(c.replace("en:", "").strip() for c in cls.split(",")) if cls else None
        desc = _en(e.get("description"))
        efsa_date = _en(e.get("efsa_evaluation_date"))
        efsa_url = _en(e.get("efsa_evaluation_url"))
        of_interest = _en(e.get("anses_additives_of_interest")) == "yes"
        vegan = _en(e.get("vegan"))

        sec = [f"## {title}"]
        if cls:
            sec.append(f"- Function: {cls}")
        if desc:
            sec.append(f"- Description: {desc}")
        if of_interest:
            sec.append("- Flagged by ANSES (French food-safety agency) as an **additive of interest** "
                       "(one regulators are monitoring more closely).")
        if efsa_date:
            sec.append(f"- EFSA safety evaluation on record ({efsa_date})"
                       + (f": {efsa_url}" if efsa_url else "") + ".")
        else:
            sec.append("- No EFSA re-evaluation date on record in the taxonomy.")
        if vegan:
            sec.append(f"- Vegan: {vegan}.")
        sec.append("- Source: Open Food Facts additives taxonomy" +
                   (" + EFSA." if efsa_date else "."))
        lines.append("\n".join(sec) + "\n")
        n += 1

    OUT.write_text("\n".join(lines))
    print(f"Wrote {n} additive reference entries -> {OUT}")
    return n


if __name__ == "__main__":
    build()
