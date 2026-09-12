"""Tests for the SQLite storage layer."""

from __future__ import annotations

import pytest

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
from storage.models import Gap, Paper, Summary


def _make_paper(paper_id: str = "2401.12345") -> Paper:
    """Build a Paper instance with sensible defaults for tests."""
    return Paper(
        paper_id=paper_id,
        title="Test Paper",
        authors='["Alice", "Bob"]',
        year=2024,
        abstract="An abstract.",
        sections_json='{"introduction": "text"}',
        references_json='["2401.00001"]',
        parse_source="pdf",
    )


def _make_summary(paper_id: str = "2401.12345") -> Summary:
    """Build a Summary instance tied to a paper id."""
    return Summary(
        paper_id=paper_id,
        problem_statement="Problem",
        method="Method",
        dataset="Dataset",
        key_results="Results",
        limitations="Limitations",
        contributions_json='["contribution one"]',
    )


def _make_gap(gap_id: str = "g1") -> Gap:
    """Build a Gap instance for tests."""
    return Gap(
        gap_id=gap_id,
        description="A research gap.",
        type="methodology",
        evidence_paper_ids_json='["2401.12345"]',
    )


@pytest.fixture()
def db(tmp_path):
    """Initialize a fresh database for each test."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    init_db(db_url)
    yield tmp_path
    close_session()


def test_insert_and_get_paper(db):
    """An inserted paper can be retrieved and has identical values."""
    paper = _make_paper()
    insert_paper(paper)
    fetched = get_paper(paper.paper_id)
    assert fetched is not None
    assert fetched.title == paper.title
    assert fetched.year == paper.year
    assert fetched.parse_source == paper.parse_source


def test_to_dict_and_from_dict_roundtrip():
    """Paper.to_dict / from_dict preserve all fields through JSON serialization."""
    paper = _make_paper()
    data = paper.to_dict()
    rebuilt = Paper.from_dict(data)
    assert rebuilt.to_dict() == data


def test_list_papers(db):
    """Multiple inserted papers are all returned by list_papers."""
    insert_paper(_make_paper("a.1"))
    insert_paper(_make_paper("a.2"))
    assert len(list_papers()) == 2


def test_insert_and_get_summary(db):
    """An inserted summary can be retrieved by paper_id."""
    paper = _make_paper()
    insert_paper(paper)
    summary = _make_summary(paper.paper_id)
    insert_summary(summary)
    fetched = get_summary(paper.paper_id)
    assert fetched is not None
    assert fetched.method == summary.method
    assert fetched.contributions_json == summary.contributions_json


def test_list_summaries(db):
    """All inserted summaries are returned by list_summaries."""
    paper = _make_paper()
    insert_paper(paper)
    insert_summary(_make_summary(paper.paper_id))
    assert len(list_summaries()) == 1


def test_insert_and_list_gaps(db):
    """An inserted gap appears in list_gaps."""
    gap = _make_gap()
    insert_gap(gap)
    gaps = list_gaps()
    assert len(gaps) == 1
    assert gaps[0].gap_id == gap.gap_id
    assert gaps[0].type == gap.type


def test_rollback(db):
    """An exception inside get_session rolls back the transaction."""
    paper = _make_paper("rb.1")
    with pytest.raises(RuntimeError):
        with get_session() as session:
            session.add(paper)
            raise RuntimeError("boom")
    assert get_paper("rb.1") is None


def test_duplicate_paper(db):
    """Inserting the same paper_id twice raises DuplicatePaperError."""
    insert_paper(_make_paper("dup.1"))
    with pytest.raises(DuplicatePaperError):
        insert_paper(_make_paper("dup.1"))
