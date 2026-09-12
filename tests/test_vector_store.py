"""Tests for the ChromaDB vector store."""

from __future__ import annotations

import pytest

from storage.vector_store import VectorStore, VectorStoreError


@pytest.fixture()
def store(tmp_path):
    """Create a fresh VectorStore backed by a temporary directory."""
    s = VectorStore(persist_directory=str(tmp_path / "chroma"))
    s.create_collection()
    yield s
    s.reset_collection()


def test_add_and_count(store):
    """Adding a document increments the collection count."""
    result = store.add_document(
        text="hello world",
        paper_id="2401.12345",
        section="abstract",
        chunk_id="2401.12345_abs",
    )
    assert result["status"] == "added"
    assert store.count() == 1


def test_duplicate_skipped(store):
    """A second add with the same chunk_id is skipped and does not increment count."""
    store.add_document("hello", "p1", "abstract", "c1")
    result = store.add_document("hello", "p1", "abstract", "c1")
    assert result["status"] == "skipped"
    assert store.count() == 1


def test_query(store):
    """Query returns the most semantically similar document."""
    store.add_document("deep learning for nlp", "p1", "abstract", "c1")
    store.add_document("cooking recipes", "p2", "abstract", "c2")
    results = store.query(["machine learning"], n_results=1)
    assert results["ids"][0][0] == "c1"


def test_delete_document(store):
    """Deleting a document decrements the collection count."""
    store.add_document("hello", "p1", "abstract", "c1")
    store.delete_document("c1")
    assert store.count() == 0


def test_reset_collection(store):
    """Resetting the collection clears all documents."""
    store.add_document("hello", "p1", "abstract", "c1")
    store.reset_collection()
    assert store.count() == 0


def test_persist_survives_reinstantiation(tmp_path):
    """Documents survive a process restart (simulated by a new instance)."""
    persist_dir = str(tmp_path / "chroma")
    s1 = VectorStore(persist_directory=persist_dir)
    s1.create_collection()
    s1.add_document("persistent text", "p1", "abstract", "c1")
    s1.persist()

    s2 = VectorStore(persist_directory=persist_dir)
    s2.create_collection()
    assert s2.count() == 1
    results = s2.query(["persistent"], n_results=1)
    assert results["ids"][0][0] == "c1"


def test_vector_store_error_on_uninitialized_collection():
    """Querying before create_collection raises VectorStoreError."""
    s = VectorStore()
    with pytest.raises(VectorStoreError):
        s.query(["anything"])