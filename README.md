# Truman — the honest truth about what's in your food

> Point it at a food — a **photo of the ingredients label**, or just a name like
> *"McDon's Big Mac"* — and Truman tells you, in plain language, what's actually
> in it and what those ingredients mean for you, grounded in real sources.
> No hype, no fear-mongering, no guessing: if the evidence isn't there, it says so.

**🔴 Live demo: https://truman-food-agent.fly.dev/** (scales to zero when idle — first request after a pause takes a few seconds to wake).

*(Agent name is a one-line setting in `src/food_agent/config.py` — swap it anytime.)*

---

## Why it exists
Ingredient labels are a wall of unpronounceable chemicals, and "clean-looking"
foods often hide dozens of additives. Truman turns a label into a clear,
sourced explanation a non-scientist can trust.

## What it does
1. **Get the ingredients** — from a **label photo** (vision) or by **name**
   (tool-call to the Open Food Facts database of millions of products).
2. **Explain each ingredient** — retrieve grounded health context and produce a
   per-ingredient + overall verdict (clean / worth-knowing / concerning), with
   citations to the sources it used.
3. **Stay honest** — when the corpus and databases don't have solid evidence on
   something, it says *"I don't have reliable data on this,"* rather than inventing.

## Architecture (and why it's built this way)

```
 label photo ─┐                         ┌─ curated RAG corpus  (common additives,
 food name ───┼─► ingredient list ─► retrieval ┤   high-quality sourced docs)
              │      (tools)              └─ PubChem live lookup (long-tail fallback)
              │                                        │
              └────────────► TOOL-CALLING AGENT ◄──────┘
                                    │
                        grounded, cited, structured verdict
                                    │
                          eval harness + tracing
```

- **Agent + tool-calling** — the model decides when to look up a food, read a
  label, search the corpus, or hit PubChem.
- **Hybrid retrieval** — a curated vector store for common ingredients (fast,
  trustworthy) **plus** live PubChem retrieval for the long tail. Curated RAG +
  dynamic API fallback = near-complete coverage without embedding the universe.
- **Grounding first** — answers cite their sources; unknowns are admitted.
- **Evaluation** — a committed golden set + retrieval metrics (recall@k) + an
  LLM-as-judge for faithfulness/groundedness, runnable as a regression gate.
- **Observability** — every request writes a trace (steps, tools, tokens, cost).

## Engineering practices this demonstrates
Agentic tool-calling · retrieval-augmented generation (chunking, embeddings,
vector search, reranking, grounding) · hybrid static+dynamic retrieval ·
structured outputs · hallucination control / honest abstention · multimodal
(vision) · **first-class evaluation (LLM-as-judge + retrieval metrics + regression)** ·
observability/tracing · a clean FastAPI service. *(Built to the stack Applied-AI
engineering roles keep asking for.)*

## Layout
```
src/food_agent/
  config.py            # agent name, models, params
  tools/
    openfoodfacts.py   # ingredient lookup by name/barcode         [working]
    pubchem.py         # chemical long-tail lookup                 [working]
  rag/
    vectorstore.py     # cosine vector store (save/load)           [working]
    ingest.py          # chunk + embed the corpus -> index         [needs API key]
    retriever.py       # hybrid retrieve (corpus -> PubChem)       [needs API key]
  vision.py            # read ingredients off a label photo        [needs API key]
  agent.py             # tool-calling loop + grounded synthesis    [needs API key]
data/corpus/           # curated additive/ingredient source docs   (see SOURCES.md)
eval/                  # golden_set.jsonl + run_eval.py
traces/                # per-request observability traces
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your OpenAI key into .env
python -m food_agent.rag.ingest     # build the index (after adding corpus docs)
python -m food_agent.agent "mcdonalds big mac"
```

## Status
Working now (no key needed): the Open Food Facts and PubChem tools, the vector
store. Next (needs the OpenAI key): corpus ingest/embeddings, the tool-calling
agent loop, vision label-reading, and the eval harness.
