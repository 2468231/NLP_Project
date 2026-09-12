"""Persistent ChromaDB vector store wrapper for paper text chunks."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import chromadb

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when the vector store cannot be initialized or used."""


class VectorStore:
    """Persistent ChromaDB collection for paper text chunks.

    Embeddings are produced locally with the sentence-transformers
    ``all-MiniLM-L6-v2`` model and stored in a cosine-similarity collection.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "paper_chunks",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._embedding_function = None

    def _ensure_client(self) -> chromadb.PersistentClient:
        """Create the persistent Chroma client on demand."""
        if self._client is None:
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            except Exception as exc:
                raise VectorStoreError(f"Chroma unavailable: {exc}") from exc
        return self._client

    def _ensure_embedding_function(self) -> Any:
        """Load the sentence-transformers embedding function on demand."""
        if self._embedding_function is None:
            try:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            except ImportError:
                from chromadb.extensions.embedding_functions import SentenceTransformerEmbeddingFunction
            try:
                self._embedding_function = SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
            except Exception as exc:
                raise VectorStoreError(f"Unable to load embedding model: {exc}") from exc
        return self._embedding_function

    def create_collection(self) -> None:
        """Create or load the named collection with cosine similarity."""
        try:
            client = self._ensure_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._ensure_embedding_function(),
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection %s ready.", self.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Unable to create collection: {exc}") from exc

    def _collection_or_error(self):
        """Return the active collection or raise a clear error."""
        if self._collection is None:
            raise VectorStoreError("Collection not initialized. Call create_collection() first.")
        return self._collection

    def add_document(self, text: str, paper_id: str, section: str, chunk_id: str) -> dict[str, str]:
        """Add one text chunk, skipping duplicate chunk ids."""
        collection = self._collection_or_error()
        try:
            existing = collection.get(ids=[chunk_id])
            if existing and existing.get("ids"):
                return {"status": "skipped", "chunk_id": chunk_id}
            collection.add(
                documents=[text],
                metadatas=[
                    {
                        "paper_id": paper_id,
                        "section": section,
                        "chunk_id": chunk_id,
                    }
                ],
                ids=[chunk_id],
            )
            return {"status": "added", "chunk_id": chunk_id}
        except Exception as exc:
            raise VectorStoreError(f"Unable to add document {chunk_id}: {exc}") from exc

    def add_documents(
        self,
        texts: list[str],
        paper_ids: list[str],
        sections: list[str],
        chunk_ids: list[str],
    ) -> list[dict[str, str]]:
        """Add multiple text chunks, skipping duplicates."""
        if not (len(texts) == len(paper_ids) == len(sections) == len(chunk_ids)):
            raise ValueError("All input lists must have the same length.")
        return [
            self.add_document(text, paper_id, section, chunk_id)
            for text, paper_id, section, chunk_id in zip(texts, paper_ids, sections, chunk_ids)
        ]

    def query(self, query_texts: list[str], n_results: int = 5) -> dict[str, Any]:
        """Query the collection with one or more text queries."""
        collection = self._collection_or_error()
        try:
            return collection.query(query_texts=query_texts, n_results=n_results)
        except Exception as exc:
            raise VectorStoreError(f"Unable to query collection: {exc}") from exc

    def delete_document(self, chunk_id: str) -> None:
        """Delete a single chunk by id."""
        collection = self._collection_or_error()
        try:
            collection.delete(ids=[chunk_id])
        except Exception as exc:
            raise VectorStoreError(f"Unable to delete document {chunk_id}: {exc}") from exc

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        collection = self._collection_or_error()
        try:
            return collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Unable to count documents: {exc}") from exc

    def reset_collection(self) -> None:
        """Delete and recreate the collection, clearing all stored chunks."""
        client = self._ensure_client()
        try:
            client.delete_collection(name=self.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Unable to reset collection: {exc}") from exc
        self.create_collection()

    def persist(self) -> None:
        """Persist the current state to disk.

        Newer Chroma releases auto-persist; in that case ``persist`` is a
        no-op rather than a failure.
        """
        client = self._ensure_client()
        try:
            persist = getattr(client, "persist", None)
            if callable(persist):
                persist()
        except Exception as exc:
            raise VectorStoreError(f"Unable to persist collection: {exc}") from exc

    def close(self) -> None:
        """Release the Chroma client and any loaded embedding model."""
        if self._client is not None:
            try:
                persist = getattr(self._client, "persist", None)
                if callable(persist):
                    persist()
            except Exception as exc:
                logger.warning("Error during close: %s", exc)
            self._client = None
        self._collection = None
        self._embedding_function = None
