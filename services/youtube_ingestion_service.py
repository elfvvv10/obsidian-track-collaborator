"""YouTube-specific ingestion helpers."""

from __future__ import annotations

from config import AppConfig
from services.import_genre_service import ImportGenreService
from services.models import IngestionRequest, IngestionResponse
from services.video_ingestion_service import VideoIngestionService


class YouTubeIngestionService:
    """Import YouTube content into the vault as structured video knowledge notes."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.import_genre_service = ImportGenreService(config)
        self.video_ingestion_service = VideoIngestionService(config)

    def ingest(self, request: IngestionRequest) -> IngestionResponse:
        """Import a YouTube URL into the configured vault folder."""
        import_genre = self.import_genre_service.canonicalize(request.import_genre)
        output_dir = self.import_genre_service.destination_for(
            self.config.youtube_ingestion_path,
            import_genre,
        )
        return self.video_ingestion_service.ingest_youtube(
            request,
            output_dir=output_dir,
            import_genre=import_genre,
        )
