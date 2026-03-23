"""General external content ingestion orchestration."""

from __future__ import annotations

from config import AppConfig
from services.index_service import IndexService
from services.models import IngestionRequest, IngestionResponse
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
        index_service_cls: type[IndexService] = IndexService,
    ) -> None:
        self.config = config
        self.webpage_service_cls = webpage_service_cls
        self.youtube_service_cls = youtube_service_cls
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
        """Import a YouTube transcript into the vault and optionally trigger indexing."""
        youtube_service = self.youtube_service_cls(self.config)
        response = youtube_service.ingest(request)

        should_index = request.index_now if request.index_now is not None else self.config.auto_index_after_ingestion
        if should_index:
            self.index_service_cls(self.config).index(reset_store=False)
            response.index_triggered = True

        return response
