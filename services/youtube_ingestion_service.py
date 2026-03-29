"""YouTube-specific ingestion helpers."""

from __future__ import annotations

from config import AppConfig
from services.import_genre_service import ImportGenreService
from services.knowledge_category_service import KnowledgeCategoryService
from services.models import IngestionRequest, IngestionResponse
from services.video_ingestion_service import VideoIngestionService


class YouTubeIngestionService:
    """Import YouTube content into the vault as structured video knowledge notes."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.import_genre_service = ImportGenreService(config)
        self.knowledge_category_service = KnowledgeCategoryService(config)
        self.video_ingestion_service = VideoIngestionService(config)

    def ingest(self, request: IngestionRequest) -> IngestionResponse:
        """Import a YouTube URL into the configured vault folder."""
        import_genre = self.import_genre_service.canonicalize(request.import_genre)
        knowledge_category = self.knowledge_category_service.validate_or_raise(request.knowledge_category)
        output_dir = self.import_genre_service.destination_for(
            self.config.youtube_ingestion_path,
            import_genre,
        )
        request = IngestionRequest(
            source=request.source,
            title_override=request.title_override,
            index_now=request.index_now,
            import_genre=request.import_genre,
            knowledge_category=knowledge_category,
        )
        return self.video_ingestion_service.ingest_youtube(
            request,
            output_dir=output_dir,
            import_genre=import_genre,
        )
