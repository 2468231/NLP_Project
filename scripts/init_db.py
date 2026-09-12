"""Initialize the SQLite database and Chroma collection."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import init_db
from storage.vector_store import VectorStore


def main() -> None:
    """Create the SQLite schema and the persistent Chroma collection."""
    os.makedirs("data", exist_ok=True)
    init_db("sqlite:///data/nlp05.db")
    store = VectorStore(persist_directory="data/chroma")
    store.create_collection()
    print("SQLite database initialized at data/nlp05.db")
    print("Chroma collection 'paper_chunks' ready at data/chroma")


if __name__ == "__main__":
    main()
