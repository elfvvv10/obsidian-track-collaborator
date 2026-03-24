"""Tests for chat and task prompt-state injection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.models import (
    AnswerMode,
    ChatMessage,
    CollaborationWorkflow,
    RetrievalMode,
    SessionTask,
    WorkflowInput,
)
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


class ChatTodoPromptTests(unittest.TestCase):
    def test_recent_conversation_block_appears_once_for_supported_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Help me refine the drop.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                recent_conversation=[
                    ChatMessage(role="user", content="The drop still feels weak.", created_at="2026-03-24 10:00:00"),
                    ChatMessage(role="assistant", content="Try muting one support layer.", created_at="2026-03-24 10:01:00"),
                ],
            )

            self.assertIn("BEGIN RECENT CONVERSATION", payload.system_prompt)
            self.assertIn("User: The drop still feels weak.", payload.system_prompt)
            self.assertIn("Assistant: Try muting one support layer.", payload.system_prompt)
            self.assertEqual(payload.system_prompt.count("BEGIN RECENT CONVERSATION"), 1)

    def test_no_conversation_block_when_history_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Help me refine the drop.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
            )

            self.assertNotIn("BEGIN RECENT CONVERSATION", payload.system_prompt)

    def test_current_tasks_block_appears_once_for_supported_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Help me finish this section.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                current_tasks=[
                    SessionTask(
                        id="1",
                        text="Remove support layer before drop",
                        status="open",
                        source="user",
                        created_at="2026-03-24 10:00:00",
                    ),
                    SessionTask(
                        id="2",
                        text="Shorten intro by 8 bars",
                        status="completed",
                        source="assistant",
                        created_at="2026-03-24 10:01:00",
                    ),
                ],
            )

            self.assertIn("BEGIN CURRENT TASKS", payload.system_prompt)
            self.assertIn("[ ] Remove support layer before drop", payload.system_prompt)
            self.assertIn("[x] Shorten intro by 8 bars", payload.system_prompt)
            self.assertEqual(payload.system_prompt.count("BEGIN CURRENT TASKS"), 1)

    def test_no_task_block_when_tasks_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Help me finish this section.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
            )

            self.assertNotIn("BEGIN CURRENT TASKS", payload.system_prompt)

    def test_prompt_order_is_framework_then_track_context_then_tasks_then_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            project_dir = vault / "Projects" / "Moonlit Driver"
            project_dir.mkdir(parents=True)
            (root / "output").mkdir()
            framework_path = root / "track_critique_framework_v1.md"
            framework_path.write_text("Framework guidance.", encoding="utf-8")
            (project_dir / "track_context.md").write_text(
                "---\ntrack_title: Moonlit Driver\n---\n\n## Structure\n\n- Intro\n",
                encoding="utf-8",
            )
            config = make_config(root)
            config.track_critique_framework_path = str(framework_path)

            payload = PromptService(config).build_prompt_payload(
                "Help me refine this track.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                workflow_input=WorkflowInput(track_context_path="Projects/Moonlit Driver"),
                current_tasks=[
                    SessionTask(
                        id="1",
                        text="Remove support layer before drop",
                        status="open",
                        source="user",
                        created_at="2026-03-24 10:00:00",
                    )
                ],
                recent_conversation=[
                    ChatMessage(role="user", content="The drop still feels weak.", created_at="2026-03-24 10:00:00")
                ],
            )

            system_prompt = payload.system_prompt
            self.assertLess(
                system_prompt.index("BEGIN INTERNAL CRITIQUE FRAMEWORK"),
                system_prompt.index("BEGIN INTERNAL TRACK CONTEXT"),
            )
            self.assertLess(
                system_prompt.index("BEGIN INTERNAL TRACK CONTEXT"),
                system_prompt.index("BEGIN CURRENT TASKS"),
            )
            self.assertLess(
                system_prompt.index("BEGIN CURRENT TASKS"),
                system_prompt.index("BEGIN RECENT CONVERSATION"),
            )

    def test_non_supported_workflow_does_not_receive_chat_or_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payload = PromptService(make_config(root)).build_prompt_payload(
                "General question.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.GENERAL_ASK,
                current_tasks=[
                    SessionTask(
                        id="1",
                        text="Task",
                        status="open",
                        source="user",
                        created_at="2026-03-24 10:00:00",
                    )
                ],
                recent_conversation=[
                    ChatMessage(role="user", content="Previous question", created_at="2026-03-24 10:00:00")
                ],
            )

            self.assertNotIn("BEGIN CURRENT TASKS", payload.system_prompt)
            self.assertNotIn("BEGIN RECENT CONVERSATION", payload.system_prompt)
