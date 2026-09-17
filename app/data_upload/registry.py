import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIVE_DATASET_PATH = os.path.join(BASE_DIR, "data", "active_dataset.json")

# Fallback values when no dataset has been uploaded yet — matches init.sql's
# seed schema so the app behaves exactly as it did in Phases 0-6 until the
# person uploads their own data.
DEFAULT_TABLE = "product_reviews"
DEFAULT_TEXT_COLUMN = "review_text"
DEFAULT_EMBEDDING_TABLE = "review_embeddings"
DEFAULT_ID_COLUMN = "id"
DEFAULT_EMBEDDING_FK = "review_id"


def get_active_dataset() -> dict | None:
    """Returns the active dataset's metadata, or None if nothing's been uploaded (i.e. still on seed data)."""
    if os.path.exists(ACTIVE_DATASET_PATH):
        with open(ACTIVE_DATASET_PATH, "r") as f:
            return json.load(f)
    return None


def set_active_dataset(metadata: dict):
    os.makedirs(os.path.dirname(ACTIVE_DATASET_PATH), exist_ok=True)
    with open(ACTIVE_DATASET_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def clear_active_dataset():
    """Reverts to the seed data — does NOT drop the uploaded table, just stops pointing at it."""
    if os.path.exists(ACTIVE_DATASET_PATH):
        os.remove(ACTIVE_DATASET_PATH)


def active_table() -> str:
    ds = get_active_dataset()
    return ds["table"] if ds else DEFAULT_TABLE


def active_text_column() -> str | None:
    """None means this dataset has no column suitable for semantic search — SQL-only."""
    ds = get_active_dataset()
    if ds:
        return ds.get("text_column")  # may be None even when a dataset IS active
    return DEFAULT_TEXT_COLUMN


def active_embedding_table() -> str:
    ds = get_active_dataset()
    return ds["embedding_table"] if ds else DEFAULT_EMBEDDING_TABLE


def active_id_column() -> str:
    ds = get_active_dataset()
    return ds["id_column"] if ds else DEFAULT_ID_COLUMN


def active_embedding_fk() -> str:
    ds = get_active_dataset()
    return ds["embedding_fk"] if ds else DEFAULT_EMBEDDING_FK


def has_semantic_support() -> bool:
    """False for uploads where no column looked like free text — semantic/hybrid routes should be disabled."""
    return active_text_column() is not None
