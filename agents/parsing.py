"""
NLP-05 Phase 3 — Parsing Agent

This module provides a production-quality parsing agent that converts research
paper PDFs into structured ParsedPaper dictionaries. It uses GROBID as the
primary parser and PyMuPDF as a fallback when GROBID is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import fitz  # PyMuPDF

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("nlp05.parsing")
if not logger.handlers:
    handler = logging.FileHandler(LOG_DIR / "parsing.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ParsingError(Exception):
    """Raised when a PDF cannot be parsed successfully."""


class ParsedPaper:
    """Pydantic-compatible ParsedPaper dictionary output."""


def extract_sections(text: str) -> Dict[str, str]:
    """
    Split paper text into named sections using regex heuristics.

    Args:
        text: Raw text extracted from a PDF.

    Returns:
        Dictionary of section name -> text content.
    """
    section_map = {
        "introduction": re.compile(r"^\s*(intro(?:duction)?|background)\s*[:\.-]?\s*$", re.I),
        "related_work": re.compile(
            r"^\s*(related\s+work|related\s+research|background\s+work|literature\s+review)\s*[:\.-]?\s*$",
            re.I,
        ),
        "methodology": re.compile(
            r"^\s*(method(?:ology)?|methods?|approach|methodological\s+framework)\s*[:\.-]?\s*$",
            re.I,
        ),
        "results": re.compile(
            r"^\s*(results?|evaluation|experiments?|experiment(?:s|al)?|findings)\s*[:\.-]?\s*$",
            re.I,
        ),
        "conclusion": re.compile(
            r"^\s*(conclusion(?:s)?|discussion|future\s+work|future\s+directions|summary)\s*[:\.-]?\s*$",
            re.I,
        ),
    }

    sections: Dict[str, str] = {
        "introduction": "",
        "related_work": "",
        "methodology": "",
        "results": "",
        "conclusion": "",
    }

    current_section: Optional[str] = None
    lines = text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        normalized = re.sub(r"\s+", " ", stripped)
        for section_name, pattern in section_map.items():
            if pattern.match(normalized):
                current_section = section_name
                break

        if current_section is not None:
            sections[current_section] = (
                sections[current_section] + "\n" + normalized
                if sections[current_section]
                else normalized
            )

    return sections


def extract_references(text: str) -> List[Dict[str, Any]]:
    """
    Extract references from bibliography text using regex heuristics.

    Args:
        text: Raw text extracted from a PDF.

    Returns:
        List of reference dictionaries with ref_id, raw_text, title, year.
    """
    references: List[Dict[str, Any]] = []
    lines = text.splitlines()

    # Find bibliography heading
    bib_start = -1
    for i, line in enumerate(lines):
        if re.match(r"^\s*(bibliograph(?:y|ic)|references?|works\s+cited)\s*[:\.-]?\s*$", line, re.I):
            bib_start = i
            break

    if bib_start == -1:
        return references

    # Collect bibliography lines
    bib_lines = []
    for line in lines[bib_start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        # Stop at another major section
        if re.match(r"^\s*(acknowledg|references?|bibliograph|appendix)\s*[:\.-]?\s*$", line, re.I):
            break
        bib_lines.append(stripped)

    # Parse each bibliography entry
    ref_pattern = re.compile(
        r"^(?P<author>[A-Z][A-Za-z]+)\s+"
        r"(?P<year>(?:19|20)\d{2}),?\s+"
        r"(?P<title>.*?)(?:\s+\((?P<extra>[^)]*)\))?$"
    )

    for line in bib_lines:
        match = ref_pattern.match(line)
        if not match:
            continue

        ref_id = f"R{len(references) + 1}"
        title = match.group("title").strip()
        year = match.group("year")
        author = match.group("author").strip()

        references.append(
            {
                "ref_id": ref_id,
                "raw_text": line,
                "title": title,
                "year": int(year) if year else None,
            }
        )

    return references


class ParsingAgent:
    """
    Production-quality parsing agent for NLP-05.

    Uses GROBID as primary parser and PyMuPDF as fallback.
    """

    GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

    def __init__(self) -> None:
        """Initialize the parsing agent."""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NLP-05-ParsingAgent/1.0"})

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a single PDF into a ParsedPaper dictionary.

        Args:
            pdf_path: Path to the input PDF.

        Returns:
            ParsedPaper dictionary.

        Raises:
            ParsingError: If the PDF cannot be parsed.
        """
        start_time = time.time()
        path = Path(pdf_path)

        if not path.exists():
            raise ParsingError(f"PDF does not exist: {pdf_path}")

        try:
            # Try GROBID first
            try:
                parsed = self.parse_with_grobid(str(path))
                source = "grobid"
            except Exception as e:
                logger.warning(f"GROBID failed for {path.name}: {e}")
                parsed = self.parse_with_pymupdf(str(path))
                source = "pymupdf_fallback"

            # Ensure output schema is complete
            parsed = self._ensure_schema(parsed)

            # Save JSON output
            output_dir = Path("corpus/parsed")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{parsed['paper_id']}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            # Insert into database
            self._insert_into_database(parsed)

            logger.info(
                f"PDF parsed successfully: {path.name} | source={source} | time={time.time() - start_time:.2f}s"
            )
            return parsed

        except Exception as e:
            logger.error(f"Failed to parse {path.name}: {e}", exc_info=True)
            raise ParsingError(f"Failed to parse {path.name}: {e}") from e

    def parse_directory(self, directory: str) -> Dict[str, int]:
        """
        Parse all PDFs in a directory.

        Args:
            directory: Directory containing PDF files.

        Returns:
            Summary report with success, fallback, failed counts.
        """
        report = {"success": 0, "fallback": 0, "failed": 0}
        path = Path(directory)

        if not path.exists():
            raise ParsingError(f"Directory does not exist: {directory}")

        pdf_files = sorted(path.glob("*.pdf"))
        for pdf_file in pdf_files:
            try:
                parsed = self.parse_pdf(str(pdf_file))
                report["success"] += 1
                # Determine if fallback was used by checking parse_source
                if parsed.get("parse_source") == "pymupdf_fallback":
                    report["fallback"] += 1
            except Exception as e:
                report["failed"] += 1
                logger.error(f"Skipping {pdf_file.name}: {e}")

        return report

    def parse_with_grobid(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse PDF using GROBID HTTP API.

        Args:
            pdf_path: Path to the input PDF.

        Returns:
            ParsedPaper dictionary from GROBID TEI XML.

        Raises:
            ParsingError: If GROBID is unavailable or XML is invalid.
        """
        try:
            with open(pdf_path, "rb") as f:
                response = self.session.post(
                    self.GROBID_URL,
                    files={"input": ("document.pdf", f, "application/pdf")},
                    timeout=60,
                )
            response.raise_for_status()

            xml_text = response.text
            parsed = self._parse_grobid_xml(xml_text)
            return parsed

        except requests.exceptions.Timeout as e:
            raise ParsingError(f"GROBID network timeout: {e}") from e
        except requests.exceptions.HTTPError as e:
            raise ParsingError(f"GROBID HTTP error: {e}") from e
        except Exception as e:
            raise ParsingError(f"GROBID parse error: {e}") from e

    def parse_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse PDF using PyMuPDF fallback.

        Args:
            pdf_path: Path to the input PDF.

        Returns:
            ParsedPaper dictionary.

        Raises:
            ParsingError: If PDF is malformed or scanned.
        """
        try:
            document = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in document)
            document.close()

            if len(text.strip()) < 200:
                raise ParsingError("PDF body text is under 200 characters (likely scanned or empty)")

            paper_id = Path(pdf_path).stem
            sections = extract_sections(text)
            references = extract_references(text)

            return {
                "paper_id": paper_id,
                "title": self._extract_title(text),
                "authors": self._extract_authors(text),
                "year": self._extract_year(text),
                "abstract": self._extract_abstract(text),
                "sections": sections,
                "references": references,
                "parse_source": "pymupdf_fallback",
            }

        except ParsingError:
            raise
        except Exception as e:
            raise ParsingError(f"PyMuPDF parse error: {e}") from e

    def _parse_grobid_xml(self, xml_text: str) -> Dict[str, Any]:
        """
        Parse GROBID TEI XML into ParsedPaper dictionary.

        Args:
            xml_text: TEI XML response from GROBID.

        Returns:
            ParsedPaper dictionary.

        Raises:
            ParsingError: If XML is malformed.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ParsingError(f"Invalid GROBID XML: {e}") from e

        paper_id = self._extract_paper_id_from_xml(root)
        title = self._extract_title_from_xml(root)
        authors = self._extract_authors_from_xml(root)
        abstract = self._extract_abstract_from_xml(root)
        sections = self._extract_sections_from_xml(root)
        references = self._extract_references_from_xml(root)

        year = self._extract_year_from_text(title)
        return {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "sections": sections,
            "references": references,
            "parse_source": "grobid",
        }

    def _extract_paper_id_from_xml(self, root: ET.Element) -> str:
        """Extract paper ID from TEI XML."""
        for elem in root.iter():
            if elem.tag.endswith("docId") or elem.tag.endswith("idno"):
                value = elem.text or ""
                if value:
                    return value
        return "unknown"

    def _extract_title_from_xml(self, root: ET.Element) -> str:
        """Extract title from TEI XML."""
        for elem in root.iter():
            if elem.tag.endswith("title"):
                value = " ".join(elem.itertext()).strip()
                if value:
                    return value
        return ""

    def _extract_authors_from_xml(self, root: ET.Element) -> List[str]:
        """Extract authors from TEI XML."""
        authors: List[str] = []
        for elem in root.iter():
            if elem.tag.endswith("author") or elem.tag.endswith("persName"):
                value = " ".join(elem.itertext()).strip()
                if value and value not in authors:
                    authors.append(value)
        return authors

    def _extract_abstract_from_xml(self, root: ET.Element) -> str:
        """Extract abstract from TEI XML."""
        for elem in root.iter():
            if elem.tag.endswith("abstract"):
                return " ".join(elem.itertext()).strip()
        return ""

    def _extract_sections_from_xml(self, root: ET.Element) -> Dict[str, str]:
        """Extract sections from TEI XML body."""
        sections: Dict[str, str] = {
            "introduction": "",
            "related_work": "",
            "methodology": "",
            "results": "",
            "conclusion": "",
        }

        for elem in root.iter():
            if elem.tag.endswith("div"):
                type_attr = elem.attrib.get("type", "")
                title = ""
                for child in elem:
                    if child.tag.endswith("head"):
                        title = " ".join(child.itertext()).strip()
                        break

                content = " ".join(elem.itertext()).strip()
                if not content:
                    continue

                section_name = self._normalize_section_name(type_attr, title)
                if section_name in sections:
                    sections[section_name] = content

        return sections

    def _extract_references_from_xml(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract references from TEI XML biblStruct."""
        references: List[Dict[str, Any]] = []
        for elem in root.iter():
            if elem.tag.endswith("biblStruct"):
                raw_text = " ".join(elem.itertext()).strip()
                title = ""
                year = None
                for child in elem.iter():
                    if child.tag.endswith("title") and not title:
                        title = " ".join(child.itertext()).strip()
                    if child.tag.endswith("date") and year is None:
                        year_text = child.attrib.get("when", "")
                        match = re.search(r"(?:19|20)\d{2}", year_text)
                        year = int(match.group(0)) if match else None

                references.append(
                    {
                        "ref_id": f"R{len(references) + 1}",
                        "raw_text": raw_text,
                        "title": title,
                        "year": year,
                    }
                )
        return references

    def _normalize_section_name(self, type_attr: str, title: str) -> str:
        """Normalize section names to standard names."""
        name = (type_attr or title).lower().strip()
        if "intro" in name or name == "background":
            return "introduction"
        if "related" in name or "literature" in name or "background" in name:
            return "related_work"
        if "method" in name or "approach" in name:
            return "methodology"
        if "result" in name or "evaluation" in name or "experiment" in name:
            return "results"
        if "conclu" in name or "discussion" in name or "future" in name:
            return "conclusion"
        return None

    def _extract_title(self, text: str) -> str:
        """Extract title from raw text (first non-empty line)."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) > 2:
                return stripped
        return ""

    def _extract_authors(self, text: str) -> List[str]:
        """Extract authors from raw text (line between title and abstract)."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return []
        # Common patterns: "Author, Author" or "(Author, Author)"
        line = lines[1]
        if line.startswith("(") and line.endswith(")"):
            line = line[1:-1]
        return [a.strip() for a in line.split(",") if a.strip()]

    def _extract_year(self, text: str) -> int:
        """Extract year from raw text."""
        match = re.search(r"(?:19|20)\d{2}", text[:2000])
        return int(match.group(0)) if match else None

    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract year from text (for GROBID title fallback)."""
        match = re.search(r"(?:19|20)\d{2}", text)
        return int(match.group(0)) if match else None

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from raw text."""
        match = re.search(r"\babstract\b\s*:\s*(.*?)(?=\n\s*(?:[A-Z][A-Za-z\s]+[:\.-]\s*$|\n\s*$))", text, re.I | re.S)
        if match:
            return match.group(1).strip()
        return ""

    def _ensure_schema(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure parsed output conforms to ParsedPaper schema."""
        return {
            "paper_id": str(parsed.get("paper_id", "unknown")),
            "title": str(parsed.get("title", "")),
            "authors": list(parsed.get("authors", [])),
            "year": parsed.get("year") or None,
            "abstract": str(parsed.get("abstract", "")),
            "sections": {
                "introduction": str(parsed.get("sections", {}).get("introduction", "")),
                "related_work": str(parsed.get("sections", {}).get("related_work", "")),
                "methodology": str(parsed.get("sections", {}).get("methodology", "")),
                "results": str(parsed.get("sections", {}).get("results", "")),
                "conclusion": str(parsed.get("sections", {}).get("conclusion", "")),
            },
            "references": list(parsed.get("references", [])),
            "parse_source": str(parsed.get("parse_source", "unknown")),
        }

    def _insert_into_database(self, parsed: Dict[str, Any]) -> None:
        """Insert parsed paper into SQLite database, skipping duplicates."""
        from storage.db import DuplicatePaperError, get_session, init_db
        from storage.models import Paper

        try:
            init_db("sqlite:///data/nlp05.db")
            paper = Paper(
                paper_id=parsed["paper_id"],
                title=parsed["title"],
                authors=json.dumps(parsed["authors"]),
                year=parsed["year"] if parsed.get("year") is not None else 0,
                abstract=parsed["abstract"],
                sections_json=json.dumps(parsed["sections"]),
                references_json=json.dumps(parsed["references"]),
                parse_source=parsed["parse_source"],
            )
            with get_session() as session:
                session.add(paper)
                session.flush()
        except DuplicatePaperError:
            logger.info(f"Skipping duplicate paper: {parsed.get('paper_id')}")
        except Exception as e:
            logger.warning(f"Failed to insert into database: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse PDF papers into ParsedPaper JSON")
    parser.add_argument("pdf_path", help="Path to PDF file to parse")
    args = parser.parse_args()

    agent = ParsingAgent()
    try:
        result = agent.parse_pdf(args.pdf_path)
        print(json.dumps(result, indent=2))
    except ParsingError as e:
        print(f"Error: {e}")
