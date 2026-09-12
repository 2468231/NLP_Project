"""SQLAlchemy declarative models for the NLP-05 storage layer.

Each model maps directly to a table in the SQLite schema and exposes
``to_dict`` / ``from_dict`` helpers that serialize JSON-backed columns
into native Python objects automatically.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import Column, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _parse_json(value: str | None) -> Any:
    """Parse a JSON string into a Python object, returning the value unchanged if not JSON."""
    if value is None:
        return None
    if isinstance(value, (list, dict, int, float, bool)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _serialize_json(value: Any) -> str:
    """Serialize a Python object into a JSON string for storage."""
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    return json.dumps(value)


class Base(DeclarativeBase):
    """Declarative base class shared by every model."""


class Paper(Base):
    """A parsed research paper."""

    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[str] = mapped_column(Text, nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    sections_json: Mapped[str] = mapped_column(Text, nullable=False)
    references_json: Mapped[str] = mapped_column(Text, nullable=False)
    parse_source: Mapped[str] = mapped_column(Text, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a Pydantic-compatible dict with JSON columns parsed."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": _parse_json(self.authors),
            "year": self.year,
            "abstract": self.abstract,
            "sections_json": _parse_json(self.sections_json),
            "references_json": _parse_json(self.references_json),
            "parse_source": self.parse_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        """Construct a Paper from a Pydantic-compatible dict."""
        return cls(
            paper_id=data["paper_id"],
            title=data["title"],
            authors=_serialize_json(data["authors"]),
            year=data["year"],
            abstract=data["abstract"],
            sections_json=_serialize_json(data["sections_json"]),
            references_json=_serialize_json(data["references_json"]),
            parse_source=data["parse_source"],
        )


class Summary(Base):
    """A structured summary of a paper."""

    __tablename__ = "summaries"

    paper_id: Mapped[str] = mapped_column(Text, primary_key=True)
    problem_statement: Mapped[str] = mapped_column(Text, nullable=False)
    method: Mapped[str] = mapped_column(Text, nullable=False)
    dataset: Mapped[str] = mapped_column(Text, nullable=False)
    key_results: Mapped[str] = mapped_column(Text, nullable=False)
    limitations: Mapped[str] = mapped_column(Text, nullable=False)
    contributions_json: Mapped[str] = mapped_column(Text, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a Pydantic-compatible dict with JSON columns parsed."""
        return {
            "paper_id": self.paper_id,
            "problem_statement": self.problem_statement,
            "method": self.method,
            "dataset": self.dataset,
            "key_results": self.key_results,
            "limitations": self.limitations,
            "contributions_json": _parse_json(self.contributions_json),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Summary:
        """Construct a Summary from a Pydantic-compatible dict."""
        return cls(
            paper_id=data["paper_id"],
            problem_statement=data["problem_statement"],
            method=data["method"],
            dataset=data["dataset"],
            key_results=data["key_results"],
            limitations=data["limitations"],
            contributions_json=_serialize_json(data["contributions_json"]),
        )


class Gap(Base):
    """A research gap identified in the literature."""

    __tablename__ = "gaps"

    gap_id: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_paper_ids_json: Mapped[str] = mapped_column(Text, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a Pydantic-compatible dict with JSON columns parsed."""
        return {
            "gap_id": self.gap_id,
            "description": self.description,
            "type": self.type,
            "evidence_paper_ids_json": _parse_json(self.evidence_paper_ids_json),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Gap:
        """Construct a Gap from a Pydantic-compatible dict."""
        return cls(
            gap_id=data["gap_id"],
            description=data["description"],
            type=data["type"],
            evidence_paper_ids_json=_serialize_json(data["evidence_paper_ids_json"]),
        )
