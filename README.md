# DM Your Data

A hybrid RAG system for querying tabular data in natural language that combines
LLM-generated SQL (precise aggregations, PostgreSQL) with vector-based semantic
search (pgvector) for conceptual queries.

## Phase 0 Setup

### Pull local LLMs via Ollama

```bash
ollama pull llama3.2:3b        # fast model, used for query routing (Phase 3)
ollama pull qwen2.5-coder:7b   # stronger model, used for SQL generation (Phase 1)
```

Swap these for whatever fits your hardware — `qwen2.5-coder` in particular is
worth keeping since it's tuned for code/SQL generation.

### Verify everything is wired up

```bash
python scripts/test_connection.py
```

You should see OK for Postgres, schema introspection, and Ollama. Fix anything
that fails before moving to Phase 1 — the whole pipeline depends on this
foundation.

## Project Structure

```
dm-your-data/
├── app/
│   ├── __init__.py       # Flask app factory
│   ├── routes.py         # routes (health check, schema introspection)
│   ├── db/
│   │   └── connection.py # DB engine + schema introspection
│   └── templates/
│       └── index.html
├── scripts/
│   ├── init.sql              # DB schema + seed data
│   ├── setup_db.sh           # creates role/DB, enables pgvector, runs init.sql
│   └── test_connection.py    # sanity check script
├── data/                  # place datasets here
├── tests/                 # test suite (Phase 5 eval harness lands here)
├── requirements.txt
├── .env.example
└── run.py
```

## Roadmap

- [x] Phase 0 — Setup & scoping
- [x] Phase 1 — Core text-to-SQL pipeline
- [x] Phase 2 — Semantic search layer
- [x] Phase 3 — Query router
- [x] Phase 4 — Data science layer (profiling, outlier detection)
- [x] Phase 5 — Evaluation harness
- [ ] Phase 6 — Frontend/UX polish
- [ ] Phase 7 — Documentation & report