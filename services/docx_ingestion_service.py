"""DOCX-specific ingestion helpers."""

from __future__ import annotations

from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

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


_WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_CORE_NS = {"dc": "http://purl.org/dc/elements/1.1/"}


class DocxIngestionService:
    """Import local DOCX files into the vault as Markdown notes."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.import_genre_service = ImportGenreService(config)
        self.knowledge_category_service = KnowledgeCategoryService(config)

    def ingest(self, request: IngestionRequest) -> IngestionResponse:
        """Import a local DOCX file into the configured vault folder."""
        docx_path = Path(request.source).expanduser().resolve()
        _validate_docx_path(docx_path)

        extracted = self._extract(docx_path)
        title = request.title_override.strip() if request.title_override else extracted["title"]
        if not title:
            title = fallback_title_from_path(docx_path, default_name="word-document")

        import_genre = self.import_genre_service.canonicalize(request.import_genre)
        knowledge_category = self.knowledge_category_service.validate_or_raise(request.knowledge_category)
        output_dir = self.import_genre_service.destination_for(
            self.config.docx_ingestion_path,
            import_genre,
        )
        ensure_directory(output_dir)
        destination = make_ingestion_destination(output_dir, title)

        body = build_ingested_markdown_note(
            title=title,
            source_type="docx_import",
            source_path=docx_path.as_posix(),
            content_heading="Extracted Content",
            content=extracted["content"],
            status="imported",
            indexed=False,
            extra_frontmatter={
                "genre": import_genre,
                "knowledge_category": knowledge_category or "",
                "schema_version": "docx_import_v1",
            },
            extra_metadata_lines=[
                ("Genre", import_genre),
                ("Knowledge Category", knowledge_category or ""),
                ("Original Filename", docx_path.name),
            ],
        )
        destination.write_text(body, encoding="utf-8")

        return IngestionResponse(
            source=docx_path.as_posix(),
            source_type="docx",
            saved_path=destination,
            title=title,
            import_genre=import_genre,
            knowledge_category=knowledge_category,
            warnings=extracted["warnings"],
        )

    def _extract(self, docx_path: Path) -> dict[str, object]:
        try:
            with ZipFile(docx_path) as archive:
                document_xml = archive.read("word/document.xml")
                core_xml = archive.read("docProps/core.xml") if "docProps/core.xml" in archive.namelist() else None
        except KeyError as exc:
            raise RuntimeError(f"DOCX ingestion could not find required document data in {docx_path.name}.") from exc
        except BadZipFile as exc:
            raise RuntimeError(f"DOCX ingestion could not read {docx_path.name} as a valid .docx file.") from exc

        title = _extract_docx_title(core_xml) if core_xml is not None else ""
        content = _extract_docx_text(document_xml)
        warnings: list[str] = []
        if not content:
            warnings.append("DOCX extraction returned very little readable text.")
            content = "Very little readable text could be extracted from this DOCX file."
        return {"title": title, "content": content, "warnings": warnings}


def _validate_docx_path(docx_path: Path) -> None:
    if not docx_path.exists() or not docx_path.is_file():
        raise ValueError(f"DOCX ingestion requires an existing local file: {docx_path}")
    if docx_path.suffix.lower() != ".docx":
        raise ValueError("DOCX ingestion only supports local .docx files in this phase.")


def _extract_docx_title(core_xml: bytes) -> str:
    try:
        root = ET.fromstring(core_xml)
    except ET.ParseError:
        return ""
    title_node = root.find("dc:title", _CORE_NS)
    return _clean_text(title_node.text if title_node is not None else "")


def _extract_docx_text(document_xml: bytes) -> str:
    try:
        root = ET.fromstring(document_xml)
    except ET.ParseError as exc:
        raise RuntimeError("DOCX ingestion could not parse the document XML.") from exc

    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", _WORD_NS):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", _WORD_NS)]
        cleaned = _clean_text("".join(texts))
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()
