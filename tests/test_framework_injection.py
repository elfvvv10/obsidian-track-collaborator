"""Tests for workflow framework loading and prompt injection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import AppConfig
from services.framework_service import FrameworkService
from services.models import AnswerMode, CollaborationWorkflow, RetrievalMode, WorkflowInput
from services.prompt_service import PromptService


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


class FrameworkInjectionTests(unittest.TestCase):
    def test_track_concept_critique_injects_framework_into_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            framework_path = root / "track_critique_framework_v1.md"
            framework_path.write_text("Focus on finishability and direct, producer-grade critique.", encoding="utf-8")
            config = make_config(root)
            config.track_critique_framework_path = str(framework_path)

            payload = PromptService(config).build_prompt_payload(
                "Critique this track concept.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                workflow_input=WorkflowInput(genre="melodic techno"),
            )

            self.assertIn("BEGIN INTERNAL CRITIQUE FRAMEWORK", payload.system_prompt)
            self.assertIn("Focus on finishability", payload.system_prompt)
            self.assertIn("END INTERNAL CRITIQUE FRAMEWORK", payload.system_prompt)
            self.assertEqual(payload.system_prompt.count("BEGIN INTERNAL CRITIQUE FRAMEWORK"), 1)
            self.assertEqual(payload.system_prompt.count("END INTERNAL CRITIQUE FRAMEWORK"), 1)
            self.assertNotIn("BEGIN INTERNAL CRITIQUE FRAMEWORK", payload.user_prompt)

    def test_non_critique_workflow_does_not_inject_framework(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            framework_path = root / "track_critique_framework_v1.md"
            framework_path.write_text("This should only apply to critique.", encoding="utf-8")
            config = make_config(root)
            config.track_critique_framework_path = str(framework_path)

            payload = PromptService(config).build_prompt_payload(
                "Plan this arrangement.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
            )

            self.assertNotIn("BEGIN INTERNAL CRITIQUE FRAMEWORK", payload.system_prompt)
            self.assertIn("Turn the idea into a practical section-by-section arrangement plan.", payload.user_prompt)

    def test_missing_framework_file_falls_back_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            config.track_critique_framework_path = str(root / "missing_framework.md")

            class MissingFrameworkService:
                def __init__(self, config: AppConfig) -> None:
                    del config

                def get_framework_text(self, workflow, domain_profile) -> str:
                    del workflow, domain_profile
                    return ""

            payload = PromptService(config, framework_service_cls=MissingFrameworkService).build_prompt_payload(
                "Critique this track concept.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
            )

            self.assertNotIn("BEGIN INTERNAL CRITIQUE FRAMEWORK", payload.system_prompt)
            self.assertIn("Active collaboration workflow: track_concept_critique.", payload.system_prompt)

    def test_override_path_takes_precedence_over_repo_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_root = root / "repo"
            (repo_root / "templates" / "frameworks").mkdir(parents=True)
            (repo_root / "templates" / "frameworks" / "track_critique_framework_v1.md").write_text(
                "Default framework text.",
                encoding="utf-8",
            )
            override_path = root / "override_framework.md"
            override_path.write_text("Override framework text.", encoding="utf-8")
            config = make_config(root)
            config.track_critique_framework_path = str(override_path)

            framework_text = FrameworkService(config, repo_root=repo_root).get_framework_text(
                CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                domain_profile=configured_domain(),
            )

            self.assertEqual(framework_text, "Override framework text.")

    def test_repo_default_framework_is_used_when_override_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_root = root / "repo"
            (repo_root / "templates" / "frameworks").mkdir(parents=True)
            (repo_root / "templates" / "frameworks" / "track_critique_framework_v1.md").write_text(
                "Default framework text.",
                encoding="utf-8",
            )
            config = make_config(root)

            framework_text = FrameworkService(config, repo_root=repo_root).get_framework_text(
                CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                domain_profile=configured_domain(),
            )

            self.assertEqual(framework_text, "Default framework text.")

    def test_vault_sources_framework_is_used_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            (vault / "Sources" / "Frameworks" / "Music Production").mkdir(parents=True)
            (
                vault
                / "Sources"
                / "Frameworks"
                / "Music Production"
                / "track_critique_framework_v1.md"
            ).write_text(
                "Vault sources framework text.",
                encoding="utf-8",
            )
            config = make_config(root)

            framework_text = FrameworkService(config).get_framework_text(
                CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                domain_profile=configured_domain(),
            )

            self.assertEqual(framework_text, "Vault sources framework text.")

    def test_legacy_repo_framework_path_is_still_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_root = root / "repo"
            (repo_root / "knowledge" / "frameworks").mkdir(parents=True)
            (repo_root / "knowledge" / "frameworks" / "track_critique_framework_v1.md").write_text(
                "Legacy framework text.",
                encoding="utf-8",
            )
            config = make_config(root)

            framework_text = FrameworkService(config, repo_root=repo_root).get_framework_text(
                CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                domain_profile=configured_domain(),
            )

            self.assertEqual(framework_text, "Legacy framework text.")

    def test_framework_text_is_cached_by_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_root = root / "repo"
            framework_dir = repo_root / "templates" / "frameworks"
            framework_dir.mkdir(parents=True)
            framework_path = framework_dir / "track_critique_framework_v1.md"
            framework_path.write_text("Cached framework text.", encoding="utf-8")
            config = make_config(root)

            service = FrameworkService(config, repo_root=repo_root)
            with mock.patch("pathlib.Path.read_text", autospec=True, return_value="Cached framework text.") as read_text:
                first = service.get_framework_text(
                    CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                    domain_profile=configured_domain(),
                )
                second = service.get_framework_text(
                    CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                    domain_profile=configured_domain(),
                )

            self.assertEqual(first, "Cached framework text.")
            self.assertEqual(second, "Cached framework text.")
            self.assertEqual(read_text.call_count, 1)


def configured_domain():
    from services.models import DomainProfile

    return DomainProfile.ELECTRONIC_MUSIC
