"""Tests for optional web search behavior."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from config import AppConfig
from services.models import QueryRequest, QueryResponse
from services.query_service import QueryService
from utils import AnswerResult, RetrievedChunk
from web_search import WebSearchResult
from web_search import DuckDuckGoWebSearchClient, WikipediaWebSearchClient


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="hermes3",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=3,
    )


class Phase5WebSearchTests(unittest.TestCase):
    def test_local_only_does_not_call_web_search(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Local note",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[WebSearchResult(title="External", url="https://example.com", snippet="External info")],
        )

        response = service.ask(QueryRequest(question="agents?", retrieval_mode="local_only"))

        self.assertEqual(tracking["web_calls"], 0)
        self.assertFalse(response.web_used)
        self.assertTrue(all(source.startswith("[Local]") for source in response.sources))

    def test_auto_mode_falls_back_to_web_for_weak_local_results(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Weak local note",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.92,
                )
            ],
            web_results=[WebSearchResult(title="External", url="https://example.com", snippet="External info")],
        )

        response = service.ask(QueryRequest(question="agents?", retrieval_mode="auto"))

        self.assertEqual(tracking["web_calls"], 1)
        self.assertTrue(response.web_used)
        self.assertEqual(response.debug.retrieval_mode_used, "auto_with_web")
        self.assertTrue(response.debug.local_retrieval_weak)

    def test_hybrid_mode_uses_web_even_with_local_results(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Strong local note",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.05,
                )
            ],
            web_results=[WebSearchResult(title="External", url="https://example.com", snippet="External info")],
        )

        response = service.ask(QueryRequest(question="agents?", retrieval_mode="hybrid"))

        self.assertEqual(tracking["web_calls"], 1)
        self.assertTrue(response.web_used)
        self.assertIn("[Local] Agents (agents.md)", response.sources)
        self.assertIn("[Web] External (https://example.com)", response.sources)

    def test_web_failures_surface_as_warnings_without_breaking_local_answer(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Strong local note",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.05,
                )
            ],
            web_results=[],
            web_error=RuntimeError("search offline"),
        )

        response = service.ask(QueryRequest(question="agents?", retrieval_mode="hybrid"))

        self.assertEqual(tracking["web_calls"], 1)
        self.assertEqual(response.answer, "Answer using 1 local and 0 web")
        self.assertTrue(any("Web search was requested but unavailable" in warning for warning in response.warnings))
        self.assertFalse(response.web_used)

    def test_cli_passes_retrieval_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            response = QueryResponse(
                answer_result=AnswerResult(
                    answer="Grounded answer",
                    sources=["[Local] Agents (agents.md)"],
                    retrieved_chunks=[
                        RetrievedChunk(
                            text="Agent note content",
                            metadata={"note_title": "Agents", "source_path": "agents.md"},
                            distance_or_score=0.1,
                        )
                    ],
                ),
            )

            with patch("main.load_config", return_value=config), patch(
                "main.QueryService.ask",
                return_value=response,
            ) as ask_mock, patch(
                "main.prompt_to_save",
                return_value=False,
            ), patch(
                "sys.argv",
                ["main.py", "ask", "What do my notes say?", "--retrieval-mode", "hybrid"],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = main.main()

            self.assertEqual(exit_code, 0)
            request = ask_mock.call_args.args[0]
            self.assertEqual(request.retrieval_mode, "hybrid")

    def test_web_client_reports_invalid_json_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            client = DuckDuckGoWebSearchClient(config)

            class StubResponse:
                headers = {"content-type": "text/html"}
                text = "<html>not json</html>"

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, object]:
                    raise ValueError("bad json")

            with patch("web_search.requests.get", return_value=StubResponse()):
                with self.assertRaisesRegex(RuntimeError, "Response preview: <html>not json</html>"):
                    client.search("latest ai news")

    def test_web_client_accepts_javascript_content_type_with_json_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            client = DuckDuckGoWebSearchClient(config)

            class StubResponse:
                headers = {"content-type": "application/x-javascript"}
                text = '{"Heading":"AI","AbstractText":"Latest AI update","AbstractURL":"https://example.com"}'

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, object]:
                    raise ValueError("force fallback parser")

            with patch("web_search.requests.get", return_value=StubResponse()):
                results = client.search("latest ai news")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "AI")

    def test_web_client_falls_back_to_html_results_when_json_endpoint_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            client = DuckDuckGoWebSearchClient(config)

            class EmptyJsonResponse:
                headers = {"content-type": "application/x-javascript"}
                text = ""

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, object]:
                    raise ValueError("empty")

            class HtmlFallbackResponse:
                text = """
                <html>
                  <body>
                    <a class="result__a" href="https://example.com/article">Example Result</a>
                    <div class="result__snippet">This is the summary snippet.</div>
                  </body>
                </html>
                """

                def raise_for_status(self) -> None:
                    return None

            with patch(
                "web_search.requests.get",
                return_value=EmptyJsonResponse(),
            ), patch(
                "web_search.requests.post",
                return_value=HtmlFallbackResponse(),
            ):
                results = client.search("latest ai news")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "Example Result")
            self.assertEqual(results[0].url, "https://example.com/article")

    def test_wikipedia_client_returns_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            client = WikipediaWebSearchClient(config)

            class StubResponse:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, object]:
                    return {
                        "query": {
                            "search": [
                                {
                                    "title": "Artificial intelligence",
                                    "snippet": "<span>AI</span> is the field of building systems.",
                                }
                            ]
                        }
                    }

            with patch("web_search.requests.get", return_value=StubResponse()):
                results = client.search("artificial intelligence")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "Artificial intelligence")
            self.assertIn("wikipedia.org/wiki/Artificial_intelligence", results[0].url)


def make_query_service(
    *,
    local_chunks: list[RetrievedChunk],
    web_results: list[WebSearchResult],
    web_error: Exception | None = None,
) -> tuple[QueryService, dict[str, int]]:
    tracking = {"web_calls": 0}

    class StubEmbeddingClient:
        def __init__(self, config: AppConfig) -> None:
            pass

    class StubChatClient:
        def __init__(self, config: AppConfig) -> None:
            pass

        def answer_question(self, question: str, chunks, *, web_results=None, retrieval_mode="local_only") -> str:
            return f"Answer using {len(chunks)} local and {len(web_results or [])} web"

    class StubRetriever:
        def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
            pass

        def retrieve(self, query: str, filters=None, options=None):
            return list(local_chunks)

    class StubVectorStore:
        def __init__(self, config: AppConfig) -> None:
            pass

        def is_index_compatible(self) -> bool:
            return True

        def count(self) -> int:
            return 1

    class StubWebSearchService:
        def __init__(self, config: AppConfig) -> None:
            pass

        def search(self, query: str) -> list[WebSearchResult]:
            tracking["web_calls"] += 1
            if web_error is not None:
                raise web_error
            return list(web_results)
    root = Path(tempfile.mkdtemp())
    (root / "vault").mkdir()
    (root / "output").mkdir()
    config = make_config(root)
    service = QueryService(
        config,
        embedding_client_cls=StubEmbeddingClient,
        chat_client_cls=StubChatClient,
        retriever_cls=StubRetriever,
        vector_store_cls=StubVectorStore,
        web_search_service_cls=StubWebSearchService,
        capture_debug_trace=False,
    )
    return service, tracking
