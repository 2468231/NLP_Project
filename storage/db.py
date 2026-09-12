"""Database session management and CRUD helpers for the NLP-05 storage layer.

This module is the single source of truth for SQLite interactions.
It provides an engine, a session factory, and explicit insert / get / list
functions for each model, all of which commit safely and rollback on error.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Gap, Paper, Summary

logger = logging.getLogger(__name__)


class DuplicatePaperError(Exception):
    """Raised when attempting to insert a paper that already exists."""


_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def init_db(db_url: str = "sqlite:///data/nlp05.db") -> Engine:
    """Create the SQLite engine and all tables.

    Ensures the parent directory of the SQLite file exists before creating
    the engine so that the first connection never fails due to a missing
    folder.

    Args:
        db_url: SQLAlchemy database URL. Defaults to a file‑based SQLite DB.

    Returns:
        The created SQLAlchemy engine.
    """
    global _engine, _SessionLocal

    # Ensure the directory for a file‑based SQLite database exists
    if db_url.startswith("sqlite:///"):
        db_path = db_url[len("sqlite:///"):]
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    _engine = create_engine(db_url, future=True)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    logger.info("Database initialized at %s", db_url)
    return _engine


@contextmanager
def get_session() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations.

    Commits on successful exit and rolls back on any exception. The session
    is always closed when leaving the context.

    Yields:
        An active SQLAlchemy Session.
    """
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def insert_paper(paper: Paper) -> Paper:
    """Insert a new paper, raising on duplicate primary key."""
    with get_session() as session:
        try:
            session.add(paper)
            session.flush()
        except IntegrityError as exc:
            raise DuplicatePaperError(
                f"Paper with id '{paper.paper_id}' already exists."
            ) from exc
        return paper


def get_paper(paper_id: str) -> Optional[Paper]:
    """Retrieve a paper by its primary key, or ``None`` if missing."""
    with get_session() as session:
        return session.get(Paper, paper_id)


def list_papers() -> list[Paper]:
    """Return every paper stored in the database."""
    with get_session() as session:
        return list(session.execute(select(Paper)).scalars().all())


def insert_summary(summary: Summary) -> Summary:
    """Insert a new summary for a paper."""
    with get_session() as session:
        session.add(summary)
        session.flush()
        return summary


def get_summary(paper_id: str) -> Optional[Summary]:
    """Retrieve the summary for a paper, or ``None`` if missing."""
    with get_session() as session:
        return session.get(Summary, paper_id)


def list_summaries() -> list[Summary]:
    """Return every summary stored in the database."""
    with get_session() as session:
        return list(session.execute(select(Summary)).scalars().all())


def insert_gap(gap: Gap) -> Gap:
    """Insert a new research gap."""
    with get_session() as session:
        session.add(gap)
        session.flush()
        return gap


def list_gaps() -> list[Gap]:
    """Return every research gap stored in the database."""
    with get_session() as session:
        return list(session.execute(select(Gap)).scalars().all())


def close_session() -> None:
    """Dispose of the engine and clear the session factory."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
        logger.info("Database session closed.")