#!/usr/bin/env python3
"""
NLP-05 Phase 2: Corpus Fetcher

Downloads research paper PDFs from arXiv based on topic search.
Uses arXiv API and saves PDFs with paper_id as filename.
Handles duplicate downloads and resumption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import arxiv

from storage.db import DuplicatePaperError, get_session, init_db, insert_paper
from storage.models import Paper

# Configure logging
import logging

logging.basicConfig(
    "logs/corpus_fetcher.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_papers_from_arxiv(
    topic: str,
    max_results: int = 15,
    output_dir: str = "corpus/raw",
    category: str = "cs.CL",
) -> List[Dict]:
    """
    Fetch papers from arXiv matching the topic.

    Args:
        topic: Search topic/query
        max_results: Maximum number of papers to fetch
        output_dir: Directory to save PDFs
        category: arXiv category to search in

    Returns:
        List of paper metadata dictionaries
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct search query
    query = f"cat:{category} AND all:{topic}"
    logger.info(f"Searching arXiv with query: {query}")

    # Create arXiv client
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,  # Be respectful to arXiv
        num_retries=3
    )

    # Search for papers
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers_metadata = []
    downloaded_paper_ids = _get_existing_paper_ids(output_dir)

    try:
        for result in client.results(search):
            paper_id = result.get_short_id()

            # Skip if we already have this paper
            if paper_id in downloaded_paper_ids:
                logger.info(f"Skipping already downloaded paper: {paper_id}")
                continue

            # Download PDF
            pdf_path = _download_paper(result, output_dir)

            if pdf_path:
                # Extract metadata
                metadata = {
                    "paper_id": paper_id,
                    "title": result.title,
                    "authors": [str(author) for author in result.authors],
                    "year": result.published.year,
                    "pdf_path": str(pdf_path),
                }
                papers_metadata.append(metadata)
                downloaded_paper_ids.add(paper_id)
                logger.info(f"Successfully downloaded: {paper_id} - {result.title[:50]}...")

                # Break if we have enough papers
                if len(papers_metadata) >= max_results:
                    break

    except Exception as e:
        logger.error(f"Error fetching from arXiv: {e}")
        raise

    return papers_metadata


def _get_existing_paper_ids(output_dir: str) -> Set[str]:
    """Get set of paper IDs that already have already been downloaded."""
    paper_ids = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith(".pdf"):
                paper_id = filename[:-4]  # Remove .pdf extension
                paper_ids.add(paper_id)
    return paper_ids


def _download_paper(result: arxiv.Result, output_dir: str) -> Optional[Path]:
    """
    Download a single PDF from arXiv result.

    Args:
        result: arxiv.Result object
        output_dir: Directory to save PDF

    Returns:
        Path to downloaded PDF or None if failed
    """
    paper_id = result.get_short_id()
    pdf_filename = f"{paper_id}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Skip if file already exists and is not empty
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        logger.debug(f"PDF already exists: {pdf_filename}")
        return Path(pdf_path)

    try:
        # Download PDF
        result.download_paper(dirpath=output_dir, filename=pdf_filename)

        # Verify download
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            return Path(pdf_path)
        else:
            logger.warning(f"Downloaded file is empty: {pdf_filename}")
            return None

    except Exception as e:
        logger.error(f"Failed to download {paper_id}: {e}")
        # Clean up partial download if exists
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError:
                pass
        return None


def save_download_log(papers_metadata: List[Dict], log_path: str = "corpus/download_log.json"):
    """
    Save download log to JSON file.

    Args:
        papers_metadata: List of paper metadata dictionaries
        log_path: Path to save log file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_data = {
        "download_timestamp": datetime.now().isoformat(),
        "total_papers": len(papers_metadata),
        "papers": papers_metadata
    }

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Download log saved to: {log_path}")


def insert_papers_into_db(papers_metadata: List[Dict]):
    """
    Insert paper metadata into SQLite database from Phase 1.

    Args:
        papers_metadata: List of paper metadata dictionaries
    """
    from storage.models import Paper

    inserted_count = 0
    skipped_count = 0

    for meta in papers_metadata:
        try:
            # Create Paper object
            paper = Paper(
                paper_id=meta["paper_id"],
                title=meta["title"],
                authors=json.dumps(meta["authors"]),  # Store as JSON string
                year=meta["year"],
                abstract="",  # Will be filled by parser
                sections_json=json.dumps({}),  # Empty for now
                references_json=json.dumps([]),  # Empty for now
                parse_source="pending"  # Will be updated by parser
            )

            # Insert into DB
            with get_session() as session:
                session.add(paper)
                session.flush()

            inserted_count += 1
            logger.info(f"Inserted paper {meta['paper_id']} into database")

        except DuplicatePaperError:
            skipped_count += 1
            logger.info(f"Skipped duplicate paper: {meta['paper_id']}")
        except Exception as e:
            logger.error(f"Failed to insert paper {meta['paper_id']}: {e}")

    logger.info(f"Database insertion complete: {inserted_count} inserted, {skipped_count} skipped")


def main():
    """Main entry point for corpus fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch research paper PDFs from arXiv for NLP-05 project"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="retrieval augmented generation",
        help="Search topic for arXiv (default: 'retrieval augmented generation')"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=15,
        help="Maximum number of papers to fetch (default: 15)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="corpus/raw",
        help="Output directory for PDFs (default: corpus/raw)"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database before fetching"
    )

    args = parser.parse_args()

    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database...")
        init_db("sqlite:///data/nlp05.db")

    logger.info(f"Starting corpus fetch: topic='{args.topic}', max_papers={args.max_papers}")

    try:
        # Fetch papers from arXiv
        papers_metadata = fetch_papers_from_arxiv(
            topic=args.topic,
            max_results=args.max_papers,
            output_dir=args.output_dir
        )

        # Save download log
        save_download_log(papers_metadata)

        # Insert into database
        if papers_metadata:
            insert_papers_into_db(papers_metadata)

        logger.info(
            f"Corpus fetch completed successfully. "
            f"Fetched {len(papers_metadata)} new papers."
        )

        # Print summary for user
        print(f"\n{'='*60}")
        print(f"CORPUS FETCHER SUMMARY")
        print(f"{'='*60}")
        print(f"Topic: {args.topic}")
        print(f"Requested: {args.max_papers} papers")
        print(f"Fetched: {len(papers_metadata)} new papers")
        print(f"PDFs saved to: {args.output_dir}")
        print(f"Download log: corpus/download_log.json")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        logger.info("Corpus fetch interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Corpus fetch failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()