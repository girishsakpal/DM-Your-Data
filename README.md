# DM Your Data

A hybrid RAG system for querying tabular data in natural language that combines
LLM-generated SQL (precise aggregations, PostgreSQL) with vector-based semantic
search (pgvector) for conceptual queries.

### Pull local LLMs via Ollama

```bash
ollama pull llama3.2:3b        
ollama pull qwen2.5-coder:3b
```

### Verify everything is wired up

```bash
python scripts/test_connection.py
```

### Run the app:

```bash
python run.py
```
Then open `http://localhost:5000`.

## Numerical Dataset: Employees Data

### Sample questions to query
1. Find the average salary of employees who work remotely and have a performance rating greater than 3.
2. Find employees whose salary is greater than the average salary of their own department.
3. Calculate the average salary for each performance category.
4. Find the top 3 highest-paid employees in every city.
5. For each department, find the employee who has the highest salary among employees with a performance rating ≥ 4.
6. Find the percentage of employees in each city who are both remote workers and high performers.

## Textual Dataset: IMBD movie rating

### Sample questions to query
1. Find the percentage of reviews that are positive.
2. Find the longest review in the dataset.
3. Find reviews that contain both positive and negative expressions, such as "good" and "bad".
4. For each sentiment, calculate: Total reviews, Average length, Minimum length, Maximum length and Percentage of total reviews
5. Find reviews longer than the average review length of all reviews.


## Project Structure

```
dm-your-data/
├── app/
│   ├── __init__.py       # Flask app factory
│   ├── routes.py         # routes (health check, schema introspection, /query, /semantic-query, /ask)
│   ├── db/
│   │   └── connection.py # DB engine + schema introspection
│   ├── llm/
│   │   └── ollama_client.py   # thin wrapper around the Ollama SDK
│   ├── nlp/
│   │   ├── schema_context.py  # formats schema + sample rows for LLM prompts
│   │   ├── sql_generator.py   # prompt building + SQL extraction from LLM output
│   │   ├── sql_executor.py    # validation, safe execution, retry-on-error
│   │   ├── query_router.py    # heuristic + LLM classification (sql/semantic/hybrid)
│   │   └── hybrid.py          # decompose -> SQL pre-filter -> semantic rank
│   ├── embeddings/
│   │   ├── embedder.py         # loads/caches the local embedding model
│   │   └── semantic_search.py  # query embedding + pgvector similarity search (supports candidate_ids for hybrid)
│   ├── profiling/
│   │   ├── profiler.py         # per-column stats: nulls, cardinality, distributions
│   │   └── outliers.py         # z-score and IQR statistical outlier detection
│   ├── eval/
│   │   ├── metrics.py          # execution-accuracy comparison + precision/recall
│   │   └── runner.py           # runs the labeled dataset end-to-end, tracks history
│   ├── data_upload/
│   │   ├── registry.py         # tracks which dataset is active (upload vs. seed data)
│   │   └── ingest.py           # CSV parsing, type inference, table replacement, embedding
│   └── templates/
│       └── index.html      # the whole UI - one file, no build step
├── scripts/
│   ├── init.sql              # DB schema + seed data
│   ├── test_connection.py    # Phase 0 sanity check script
│   ├── test_phase1.py        # CLI tester for the text-to-SQL pipeline
│   ├── ingest_embeddings.py  # embeds review_text rows into review_embeddings
│   ├── test_phase2.py        # CLI tester for the semantic search pipeline
│   ├── test_phase3.py        # CLI tester for the query router (all 3 routes)
│   ├── run_profiling.py      # generates data/profile_report.json + outlier_report.json
│   ├── test_phase4.py        # CLI tester for profiling + outlier detection
│   ├── run_eval.py           # runs the full Phase 5 evaluation suite
│   └── test_upload.py        # tests the CSV upload pipeline (built-in sample or your own file)
├── tests/
│   └── eval_dataset.json     # labeled test cases (sql/semantic/hybrid) with expected answers
├── data/                  # place datasets here - also holds generated profile_report.json / outlier_report.json / eval_runs/ / eval_history.csv
├── requirements.txt
├── .env.example
└── run.py
```

## Roadmap

- [x] Phase 0 - Setup & scoping
- [x] Phase 1 - Core text-to-SQL pipeline
- [x] Phase 2 - Semantic search layer
- [x] Phase 3 - Query router
- [x] Phase 4 - Data science layer (profiling, outlier detection)
- [x] Phase 5 - Evaluation harness
- [x] Phase 6 - Frontend/UX polish
- [x] Phase 7 - Upload your own dataset feature