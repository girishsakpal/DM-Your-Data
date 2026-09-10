import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_model():
    """
    Loads the embedding model once per process. Cached because loading a
    sentence-transformers model takes a second or two — don't want to pay
    that cost on every request.
    """
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def embed_text(text: str) -> list[float]:
    """Returns a single embedding vector for one string."""
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batched version — much faster than calling embed_text in a loop for many rows."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return vectors.tolist()
