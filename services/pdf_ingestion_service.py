"""PDF-specific ingestion helpers."""

from __future__ import annotations

from pathlib import Path
import re

from config import AppConfig
from services.import_genre_service import ImportGenreService
from services.ingestion_helpers import (
    build_ingested_markdown_note,
    fallback_title_from_path,
    make_ingestion_destination,
)
from services.knowledge_category_service import KnowledgeCategoryService
from services.models import IngestionRequest, IngestionResponse
from utils import ensure_directory

try:  # pragma: no cover - optional runtime dependency
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised via fallback path
    PdfReader = None


class PdfIngestionService:
    """Import local PDF files into the vault as Markdown notes."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.import_genre_service = ImportGenreService(config)
        self.knowledge_category_service = KnowledgeCategoryService(config)

    def ingest(self, request: IngestionRequest) -> IngestionResponse:
        """Import a local PDF file into the configured vault folder."""
        pdf_path = Path(request.source).expanduser().resolve()
        _validate_pdf_path(pdf_path)

        extracted = self._extract(pdf_path)
        title = request.title_override.strip() if request.title_override else extracted["title"]
        if not title:
            title = fallback_title_from_path(pdf_path, default_name="pdf-document")

        import_genre = self.import_genre_service.canonicalize(request.import_genre)
        knowledge_category = self.knowledge_category_service.validate_or_raise(request.knowledge_category)
        output_dir = self.import_genre_service.destination_for(
            self.config.pdf_ingestion_path,
            import_genre,
        )
        ensure_directory(output_dir)
        destination = make_ingestion_destination(output_dir, title)

        body = build_ingested_markdown_note(
            title=title,
            source_type="pdf_import",
            source_path=pdf_path.as_posix(),
            content_heading="Extracted Content",
            content=extracted["content"],
            status="imported",
            indexed=False,
            extra_frontmatter={
                "genre": import_genre,
                "knowledge_category": knowledge_category or "",
                "schema_version": "pdf_import_v1",
            },
            extra_metadata_lines=[
                ("Genre", import_genre),
                ("Knowledge Category", knowledge_category or ""),
                ("Original Filename", pdf_path.name),
            ],
        )
        destination.write_text(body, encoding="utf-8")

        return IngestionResponse(
            source=pdf_path.as_posix(),
            source_type="pdf",
            saved_path=destination,
            title=title,
            import_genre=import_genre,
            knowledge_category=knowledge_category,
            warnings=extracted["warnings"],
        )

    def _extract(self, pdf_path: Path) -> dict[str, object]:
        if PdfReader is not None:
            try:
                reader = PdfReader(str(pdf_path))
                title = _clean_text((reader.metadata or {}).get("/Title", ""))
                pages: list[str] = []
                for page in reader.pages:
                    text = _clean_text(page.extract_text() or "")
                    if text:
                        pages.append(text)
                content = "\n\n".join(pages).strip()
                warnings: list[str] = []
                if not content:
                    warnings.append("PDF extraction returned very little readable text.")
                    content = "Very little readable text could be extracted from this PDF."
                return {"title": title, "content": content, "warnings": warnings}
            except Exception:
                pass

        raw_bytes = pdf_path.read_bytes()
        title = _extract_pdf_title_from_bytes(raw_bytes)
        content = _extract_pdf_text_fallback(raw_bytes)
        warnings: list[str] = []
        if not content:
            warnings.append("PDF extraction returned very little readable text.")
            content = "Very little readable text could be extracted from this PDF."
        return {"title": title, "content": content, "warnings": warnings}


def _validate_pdf_path(pdf_path: Path) -> None:
    if not pdf_path.exists() or not pdf_path.is_file():
        raise ValueError(f"PDF ingestion requires an existing local file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("PDF ingestion only supports local .pdf files in this phase.")


def _extract_pdf_title_from_bytes(raw_bytes: bytes) -> str:
    match = re.search(rb"/Title\s*\((.*?)\)", raw_bytes, flags=re.DOTALL)
    if not match:
        return ""
    return _clean_text(match.group(1).decode("latin-1", errors="ignore"))


def _extract_pdf_text_fallback(raw_bytes: bytes) -> str:
    text_fragments: list[str] = []
    decoded = raw_bytes.decode("latin-1", errors="ignore")
    for match in re.finditer(r"\((.*?)\)\s*Tj", decoded, flags=re.DOTALL):
        text = _clean_text(match.group(1).replace(r"\(", "(").replace(r"\)", ")"))
        if text:
            text_fragments.append(text)
    for match in re.finditer(r"\[(.*?)\]\s*TJ", decoded, flags=re.DOTALL):
        parts = re.findall(r"\((.*?)\)", match.group(1), flags=re.DOTALL)
        text = _clean_text(" ".join(part.replace(r"\(", "(").replace(r"\)", ")") for part in parts))
        if text:
            text_fragments.append(text)
    return "\n\n".join(_dedupe_preserve_order(text_fragments))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = " ".join(item.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()
