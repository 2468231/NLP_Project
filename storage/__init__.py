"""NLP-05 storage layer package.

Provides persistent storage for parsed papers, summaries, research gaps,
and text embeddings used by the literature review system.
"""

from storage.db import (
    DuplicatePaperError,
    close_session,
    get_paper,
    get_session,
    get_summary,
    init_db,
    insert_gap,
    insert_paper,
    insert_summary,
    list_gaps,
    list_papers,
    list_summaries,
)
from storage.models import Base, Gap, Paper, Summary
from storage.vector_store import VectorStore, VectorStoreError

__all__ = [
    "Base",
    "DuplicatePaperError",
    "Gap",
    "Paper",
    "Summary",
    "VectorStore",
    "VectorStoreError",
    "close_session",
    "get_paper",
    "get_session",
    "get_summary",
    "init_db",
    "insert_gap",
    "insert_paper",
    "insert_summary",
    "list_gaps",
    "list_papers",
    "list_summaries",
]