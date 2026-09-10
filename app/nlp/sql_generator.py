import os
import re
from app.llm.ollama_client import generate
from app.nlp.schema_context import build_schema_context

SYSTEM_PROMPT = """You are a PostgreSQL expert. Given a database schema and a \
natural language question, write a single valid PostgreSQL SELECT query that \
answers the question.

Rules:
- Output ONLY the SQL query. No explanation, no markdown code fences, no comments.
- Only generate SELECT statements. Never write INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.
- Use only the tables and columns given in the schema. Do not invent column names.
- If the question involves text similarity or "reviews about/like X", note that this \
system has a separate semantic search path — still write the best SQL approximation \
using ILIKE on the relevant text column.
- Always add a LIMIT clause (e.g. LIMIT 100) unless the question clearly asks for an \
aggregate (COUNT, AVG, SUM, etc.) that returns a single row.
"""

FEW_SHOT_EXAMPLES = """Example 1:
Question: What is the average price by category?
SQL: SELECT category, AVG(price) AS avg_price FROM product_reviews GROUP BY category;

Example 2:
Question: Show the 5 most recent reviews with a rating below 3.
SQL: SELECT * FROM product_reviews WHERE rating < 3 ORDER BY review_date DESC LIMIT 5;

Example 3:
Question: How many reviews came from the West region?
SQL: SELECT COUNT(*) FROM product_reviews WHERE region = 'West';
"""


def build_prompt(question: str, schema_context: str, previous_error: str = None, previous_sql: str = None) -> str:
    prompt = f"""Schema:
{schema_context}

{FEW_SHOT_EXAMPLES}

Question: {question}
SQL:"""

    if previous_error and previous_sql:
        # Fed back in on retry so the model can self-correct instead of repeating the same mistake.
        prompt += f"""

Note: a previous attempt produced invalid SQL. Fix it.
Previous SQL: {previous_sql}
Error: {previous_error}
Corrected SQL:"""

    return prompt


def extract_sql(raw_response: str) -> str:
    """Strips markdown fences and surrounding chatter the LLM might add despite instructions."""
    text = raw_response.strip()

    # Remove ```sql ... ``` or ``` ... ``` fences if present
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # If the model prefixed with "SQL:" anyway, strip it
    text = re.sub(r"^SQL:\s*", "", text, flags=re.IGNORECASE).strip()

    # Take up to the first semicolon-terminated statement if there's trailing chatter
    if ";" in text:
        text = text.split(";")[0] + ";"

    return text


def generate_sql(question: str, previous_error: str = None, previous_sql: str = None) -> dict:
    """
    Returns {"sql": str, "prompt": str} — prompt is included for debugging/eval logging.
    """
    model = os.getenv("SQL_MODEL", "qwen2.5-coder:7b")
    schema_context = build_schema_context()
    prompt = build_prompt(question, schema_context, previous_error, previous_sql)

    raw_response = generate(model=model, prompt=prompt, system=SYSTEM_PROMPT)
    sql = extract_sql(raw_response)

    return {"sql": sql, "prompt": prompt, "raw_response": raw_response}
