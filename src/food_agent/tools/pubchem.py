"""
Tool: look up a chemical/additive in PubChem (NIH) — the "long tail" fallback for
ingredients NOT covered by the curated RAG corpus. PubChem has ~110M compounds,
so this gives near-complete coverage of anything on a label.

Flow: name -> CID -> (formula/weight/IUPAC name) + a plain-text description with
its source. No API key required.

Design note (interview-relevant): this is DYNAMIC retrieval. The agent uses the
curated corpus first (fast, curated health context); only when coverage is thin
does it call PubChem live. Curated RAG + dynamic API fallback = agentic retrieval.
"""
from __future__ import annotations
import requests

_UA = {"User-Agent": "food-safety-agent/0.1 (portfolio project)"}
_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _cid_for_name(name: str, timeout: int) -> int | None:
    try:
        r = requests.get(f"{_BASE}/compound/name/{requests.utils.quote(name)}/cids/JSON",
                         headers=_UA, timeout=timeout)
        if r.status_code != 200:
            return None
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None
    except Exception:
        return None


def lookup_chemical(name: str, timeout: int = 15) -> dict:
    """Resolve a chemical/additive name to PubChem facts + a plain-language description."""
    cid = _cid_for_name(name, timeout)
    if not cid:
        return {"found": False, "query": name,
                "note": "Not found in PubChem — the agent should say it lacks reliable data rather than guess."}

    out = {"found": True, "source": "pubchem", "query": name, "cid": cid,
           "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"}

    # basic identity properties
    try:
        r = requests.get(f"{_BASE}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName/JSON",
                         headers=_UA, timeout=timeout)
        props = r.json()["PropertyTable"]["Properties"][0]
        out.update({"molecular_formula": props.get("MolecularFormula"),
                    "molecular_weight": props.get("MolecularWeight"),
                    "iupac_name": props.get("IUPACName")})
    except Exception:
        pass

    # plain-text description(s) with source attribution (good for grounding/citations)
    try:
        r = requests.get(f"{_BASE}/compound/cid/{cid}/description/JSON", headers=_UA, timeout=timeout)
        infos = r.json().get("InformationList", {}).get("Information", [])
        descs = [{"text": i["Description"], "source": i.get("DescriptionSourceName")}
                 for i in infos if i.get("Description")]
        out["descriptions"] = descs[:3]
    except Exception:
        out["descriptions"] = []

    return out


if __name__ == "__main__":  # quick manual test (no API key needed)
    import json, sys
    q = " ".join(sys.argv[1:]) or "aspartame"
    print(json.dumps(lookup_chemical(q), indent=2, ensure_ascii=False)[:2500])
