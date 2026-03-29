"""General external content ingestion orchestration."""

from __future__ import annotations

from config import AppConfig
from services.docx_ingestion_service import DocxIngestionService
from services.index_service import IndexService
from services.models import IngestionRequest, IngestionResponse
from services.pdf_ingestion_service import PdfIngestionService
from services.webpage_ingestion_service import WebpageIngestionService
from services.youtube_ingestion_service import YouTubeIngestionService


class IngestionService:
    """Coordinate external content import workflows and optional indexing."""

    def __init__(
        self,
        config: AppConfig,
        *,
        webpage_service_cls: type[WebpageIngestionService] = WebpageIngestionService,
        youtube_service_cls: type[YouTubeIngestionService] = YouTubeIngestionService,
        pdf_service_cls: type[PdfIngestionService] = PdfIngestionService,
        docx_service_cls: type[DocxIngestionService] = DocxIngestionService,
        index_service_cls: type[IndexService] = IndexService,
    ) -> None:
        self.config = config
        self.webpage_service_cls = webpage_service_cls
        self.youtube_service_cls = youtube_service_cls
        self.pdf_service_cls = pdf_service_cls
        self.docx_service_cls = docx_service_cls
        self.index_service_cls = index_service_cls

    def ingest_webpage(self, request: IngestionRequest) -> IngestionResponse:
        """Import a webpage into the vault and optionally trigger indexing."""
        webpage_service = self.webpage_service_cls(self.config)
        response = webpage_service.ingest(request)

        should_index = request.index_now if request.index_now is not None else self.config.auto_index_after_ingestion
        if should_index:
            self.index_service_cls(self.config).index(reset_store=False)
            response.index_triggered = True

        return response

    def ingest_youtube(self, request: IngestionRequest) -> IngestionResponse:
        """Import a YouTube video knowledge note into the vault and optionally trigger indexing."""
        youtube_service = self.youtube_service_cls(self.config)
        response = youtube_service.ingest(request)

        should_index = request.index_now if request.index_now is not None else self.config.auto_index_after_ingestion
        if should_index:
            self.index_service_cls(self.config).index(reset_store=False)
            response.index_triggered = True

        return response

    def ingest_pdf(self, request: IngestionRequest) -> IngestionResponse:
        """Import a PDF into the vault and optionally trigger indexing."""
        pdf_service = self.pdf_service_cls(self.config)
        response = pdf_service.ingest(request)

        should_index = request.index_now if request.index_now is not None else self.config.auto_index_after_ingestion
        if should_index:
            self.index_service_cls(self.config).index(reset_store=False)
            response.index_triggered = True

        return response

    def ingest_docx(self, request: IngestionRequest) -> IngestionResponse:
        """Import a DOCX document into the vault and optionally trigger indexing."""
        docx_service = self.docx_service_cls(self.config)
        response = docx_service.ingest(request)

        should_index = request.index_now if request.index_now is not None else self.config.auto_index_after_ingestion
        if should_index:
            self.index_service_cls(self.config).index(reset_store=False)
            response.index_triggered = True

        return response
