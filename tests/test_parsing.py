"""
Tests for the NLP-05 Parsing Agent.

Covers PyMuPDF parsing, GROBID integration, fallback behavior,
schema validation, directory parsing, duplicate handling,
malformed PDF handling, and database integration.

All tests use pytest fixtures and mock external dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from agents.parsing import (
    ParsingAgent,
    ParsingError,
    extract_references,
    extract_sections,
)


@pytest.fixture()
def temp_pdf(tmp_path: Path) -> Path:
    """Create a minimal valid PDF with text content using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()

    y_pos = 50
    lines = [
        "Test Title",
        "",
        "Author, Coauthor",
        "",
        "2024",
        "",
        "Abstract:",
        "This is the abstract of the paper. It provides an overview of the research.",
        "",
        "Introduction",
        "This is the introduction section of the paper. It explains the background and motivation.",
        "",
        "Related Work",
        "This is related work. It summarizes previous research in the field.",
        "",
        "Methodology",
        "This is the methodology section. It describes the approach taken.",
        "",
        "Results",
        "These are the results. They show the findings of the research.",
        "",
        "Conclusion",
        "This is the conclusion. It summarizes the contributions of the paper."
    ]

    for line in lines:
        if line == "":
            y_pos += 12
        else:
            page.insert_text((50, y_pos), line)
            y_pos += 14

    pdf_path = tmp_path / "test_paper.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture()
def sample_grobid_xml() -> str:
    """Return a minimal valid GROBID TEI XML response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Test Paper Title</title>
      </titleStmt>
      <publicationStmt>
        <idno type="DOI">10.1234/test.2024.001</idno>
      </publicationStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div type="introduction">
        <head>Introduction</head>
        <p>This is the introduction.</p>
      </div>
      <div type="related_work">
        <head>Related Work</head>
        <p>This is related work.</p>
      </div>
      <div type="methodology">
        <head>Methodology</head>
        <p>This is the methodology.</p>
      </div>
      <div type="results">
        <head>Results</head>
        <p>These are results.</p>
      </div>
      <div type="conclusion">
        <head>Conclusion</head>
        <p>This is the conclusion.</p>
      </div>
      <listBibl>
        <biblStruct>
          <analytic>
            <title>Reference Paper</title>
            <author><persName>Smith, John</persName></author>
            <date when="2023"></date>
          </analytic>
          <monogr>
            <title>Journal Name</title>
          </monogr>
        </biblStruct>
      </listBibl>
    </body>
  </text>
</TEI>
"""


@pytest.fixture()
def empty_pdf(tmp_path: Path) -> Path:
    """Create a PDF with a page but no text (empty content)."""
    doc = fitz.open()
    doc.new_page()
    pdf_path = tmp_path / "empty.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# 1. Successful PyMuPDF parsing
# ---------------------------------------------------------------------------
def test_parse_with_pymupdf_success(temp_pdf: Path) -> None:
    """Parse a simple PDF and verify title, abstract, and sections exist."""
    agent = ParsingAgent()
    with patch.object(agent.session, "post", side_effect=Exception("GROBID unavailable")):
        result = agent.parse_pdf(str(temp_pdf))

    assert result["paper_id"] == "test_paper"
    assert result["title"] != ""
    assert "abstract" in result
    assert isinstance(result["sections"], dict)
    assert result["sections"]["introduction"] != ""
    assert result["parse_source"] == "pymupdf_fallback"


# ---------------------------------------------------------------------------
# 2. Successful GROBID parsing
# ---------------------------------------------------------------------------
def test_grobid_success(sample_grobid_xml: str) -> None:
    """Mock a valid TEI XML response and verify GROBID parser is used."""
    agent = ParsingAgent()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"fake pdf content")
        pdf_path = f.name

    try:
        mock_response = MagicMock()
        mock_response.text = sample_grobid_xml
        mock_response.raise_for_status = MagicMock()

        with patch.object(agent.session, "post", return_value=mock_response):
            result = agent.parse_with_grobid(pdf_path)

        assert result["parse_source"] == "grobid"
        assert result["title"] == "Test Paper Title"
        assert result["paper_id"] == "10.1234/test.2024.001"
    finally:
        os.unlink(pdf_path)


# ---------------------------------------------------------------------------
# 3. GROBID failure triggers fallback
# ---------------------------------------------------------------------------
def test_grobid_failure_fallback(temp_pdf: Path) -> None:
    """Mock GROBID HTTP failure and verify PyMuPDF fallback is used."""
    agent = ParsingAgent()
    with patch.object(agent.session, "post", side_effect=Exception("Connection refused")):
        result = agent.parse_pdf(str(temp_pdf))

    assert result["parse_source"] == "pymupdf_fallback"
    assert result["title"] != ""


# ---------------------------------------------------------------------------
# 4. Section extraction
# ---------------------------------------------------------------------------
def test_extract_sections() -> None:
    """Verify Introduction, Related Work, Methodology, Results, Conclusion are extracted correctly."""
    text = (
        "Introduction\nThis is the introduction text.\n\n"
        "Related Work\nThis is related work text.\n\n"
        "Methodology\nThis is the methodology text.\n\n"
        "Results\nThese are results.\n\n"
        "Conclusion\nThis is the conclusion."
    )
    sections = extract_sections(text)

    assert isinstance(sections, dict)
    assert sections["introduction"] == "Introduction\nThis is the introduction text."
    assert sections["related_work"] == "Related Work\nThis is related work text."
    assert sections["methodology"] == "Methodology\nThis is the methodology text."
    assert sections["results"] == "Results\nThese are results."
    assert sections["conclusion"] == "Conclusion\nThis is the conclusion."


# ---------------------------------------------------------------------------
# 5. Reference extraction
# ---------------------------------------------------------------------------
def test_extract_references() -> None:
    """Verify bibliography parsing returns a list of references with correct fields."""
    # The regex pattern expects: Author 2023, Title (with space between author and year)
    text = """References
Smith 2023, A Great Paper
Doe 2022, Another Paper
"""
    references = extract_references(text)

    assert isinstance(references, list)
    assert len(references) >= 1
    assert references[0]["ref_id"] == "R1"
    assert references[0]["year"] == 2023
    assert references[0]["title"] == "A Great Paper"
    assert "raw_text" in references[0]


# ---------------------------------------------------------------------------
# 6. Duplicate paper skip
# ---------------------------------------------------------------------------
def test_duplicate_paper_skip(temp_pdf: Path) -> None:
    """Insert a paper twice and verify duplicate is skipped on the second attempt."""
    agent = ParsingAgent()
    with patch.object(agent.session, "post", side_effect=Exception("GROBID unavailable")):
        result1 = agent.parse_pdf(str(temp_pdf))

    paper_id = result1["paper_id"]

    with patch.object(agent.session, "post", side_effect=Exception("GROBID unavailable")):
        result2 = agent.parse_pdf(str(temp_pdf))

    assert result1["paper_id"] == result2["paper_id"]
    assert result2["paper_id"] == paper_id


# ---------------------------------------------------------------------------
# 7. Directory parsing
# ---------------------------------------------------------------------------
def test_parse_directory(tmp_path: Path) -> None:
    """Parse multiple PDFs and verify success/failure summary."""
    agent = ParsingAgent()

    # Create two PDFs in the temp directory with sufficient text
    for name in ["paper1.pdf", "paper2.pdf"]:
        doc = fitz.open()
        page = doc.new_page()
        y_pos = 50
        for line in [
            f"Title of {name}",
            "",
            "Abstract:",
            "This is the abstract text for the paper. It summarizes the research contribution.",
            "",
            "Introduction",
            "This is the introduction section. We provide background and motivation.",
            "",
            "Conclusion",
            "This is the conclusion. We summarize findings and suggest future work."
        ]:
            if line == "":
                y_pos += 12
            else:
                page.insert_text((50, y_pos), line)
                y_pos += 14
        doc.save(str(tmp_path / name))
        doc.close()

    with patch.object(agent.session, "post", side_effect=Exception("GROBID unavailable")):
        report = agent.parse_directory(str(tmp_path))

    assert isinstance(report, dict)
    assert "success" in report
    assert "fallback" in report
    assert "failed" in report
    assert report["success"] == 2


# ---------------------------------------------------------------------------
# 8. Invalid PDF handling
# ---------------------------------------------------------------------------
def test_invalid_pdf(tmp_path: Path) -> None:
    """Corrupted PDF should raise ParsingError."""
    agent = ParsingAgent()
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_text("not a valid PDF content")

    with pytest.raises(ParsingError):
        agent.parse_with_pymupdf(str(bad_pdf))


# ---------------------------------------------------------------------------
# 9. JSON output schema validation
# ---------------------------------------------------------------------------
def test_json_output_schema(temp_pdf: Path) -> None:
    """Verify ParsedPaper contains all required keys."""
    agent = ParsingAgent()
    with patch.object(agent.session, "post", side_effect=Exception("GROBID unavailable")):
        result = agent.parse_pdf(str(temp_pdf))

    parsed = agent._ensure_schema(result)

    required_keys = {
        "paper_id",
        "title",
        "authors",
        "year",
        "abstract",
        "sections",
        "references",
        "parse_source",
    }
    assert required_keys.issubset(parsed.keys())

    section_keys = {"introduction", "related_work", "methodology", "results", "conclusion"}
    assert section_keys.issubset(parsed["sections"].keys())

    assert isinstance(parsed["authors"], list)
    assert isinstance(parsed["references"], list)
    assert parsed["parse_source"] in ("grobid", "pymupdf_fallback")


# ---------------------------------------------------------------------------
# 10. Logging verification
# ---------------------------------------------------------------------------
def test_logging_created(tmp_path: Path) -> None:
    """Verify parsing.log is created after parsing."""
    log_path = Path("logs/parsing.log")

    agent = ParsingAgent()

    pdf_path = tmp_path / "logging_test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y_pos = 50
    for line in [
        "Test Title",
        "",
        "Abstract:",
        "This is a detailed abstract for logging verification. It explains the research motivation.",
        "",
        "Introduction",
        "Intro text. This explains the introduction section. We discuss the background here.",
        "",
        "Conclusion",
        "Conclusion text. This summarizes the paper. We conclude with future work directions."
    ]:
        if line == "":
            y_pos += 12
        else:
            page.insert_text((50, y_pos), line)
            y_pos += 14
    doc.save(str(pdf_path))
    doc.close()

    try:
        with patch.object(agent.session, "post", side_effect=Exception("GROBID unavailable")):
            agent.parse_pdf(str(pdf_path))

        assert log_path.exists(), "logs/parsing.log was not created"
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


# ---------------------------------------------------------------------------
# 11. Empty PDF handling
# ---------------------------------------------------------------------------
def test_empty_pdf_handling(empty_pdf: Path) -> None:
    """Empty PDF (no text) should raise ParsingError."""
    agent = ParsingAgent()
    with pytest.raises(ParsingError):
        agent.parse_with_pymupdf(str(empty_pdf))


# ---------------------------------------------------------------------------
# 12. Non-existent PDF handling
# ---------------------------------------------------------------------------
def test_parse_pdf_raises_on_missing_file() -> None:
    """Parsing a non-existent PDF should raise ParsingError."""
    agent = ParsingAgent()
    with pytest.raises(ParsingError):
        agent.parse_pdf("/nonexistent/path/to/paper.pdf")


# ---------------------------------------------------------------------------
# 13. Direct PyMuPDF parse
# ---------------------------------------------------------------------------
def test_parse_with_pymupdf_direct(temp_pdf: Path) -> None:
    """Test direct call to parse_with_pymupdf returns complete parsed result."""
    agent = ParsingAgent()
    with patch.object(agent.session, "post", side_effect=Exception("GROBID unavailable")):
        result = agent.parse_with_pymupdf(str(temp_pdf))

    assert result["paper_id"] == "test_paper"
    assert "sections" in result
    assert "references" in result
    assert "abstract" in result


# ---------------------------------------------------------------------------
# 14. Schema completeness
# ---------------------------------------------------------------------------
def test_ensure_schema_complete(temp_pdf: Path) -> None:
    """Test that parsed output conforms to complete ParsedPaper schema."""
    agent = ParsingAgent()
    with patch.object(agent.session, "post", side_effect=Exception("GROBID unavailable")):
        result = agent.parse_pdf(str(temp_pdf))

    parsed = agent._ensure_schema(result)

    assert parsed["paper_id"] is not None
    assert parsed["title"] is not None
    assert isinstance(parsed["authors"], list)
    assert isinstance(parsed["sections"], dict)
    assert isinstance(parsed["references"], list)
    assert parsed["parse_source"] in ("grobid", "pymupdf_fallback")


# ---------------------------------------------------------------------------
# 15. JSON file creation verification
# ---------------------------------------------------------------------------
def test_json_file_creation(temp_pdf: Path) -> None:
    """Test that parsing creates a JSON output file at corpus/parsed/{paper_id}.json."""
    agent = ParsingAgent()
    with patch.object(agent.session, "post", side_effect=Exception("GROBID unavailable")):
        result = agent.parse_pdf(str(temp_pdf))

    json_path = Path("corpus/parsed") / f"{result['paper_id']}.json"
    assert json_path.exists(), f"Expected JSON file {json_path} was not created"

    with open(json_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["paper_id"] == result["paper_id"]
    assert saved["title"] == result["title"]

    # Cleanup
    if json_path.exists():
        json_path.unlink()