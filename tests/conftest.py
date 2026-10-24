"""Test helpers. The scripted LLM below exists ONLY in tests: the application never
substitutes canned output for a real model response."""
from __future__ import annotations

import json
import os

import pytest

import llm_provider
from llm_provider import LLMProvider, ProviderStatus
from sample_project import sample_brief, sample_full


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch):
    """Tests must never read the developer's real .env or shell settings (API keys, provider choice).

    app.py calls `load_env_file()` and `apply_secrets()` on every run; without this, an AppTest could
    pick up a real GEMINI_API_KEY from .env or Streamlit secrets and make a real network request. Tests that need settings set them
    explicitly with monkeypatch.
    """
    for name in list(os.environ):
        if name.startswith(("FRAME_", "GEMINI_", "OLLAMA_", "GOOGLE_API")):
            monkeypatch.delenv(name)
    monkeypatch.setattr(llm_provider, "load_env_file", lambda *args, **kwargs: None)
    # Same for Streamlit secrets (a real ~/.streamlit/secrets.toml must never leak into tests).
    monkeypatch.setattr(llm_provider, "apply_secrets", lambda *args, **kwargs: [])


class ScriptedLLM(LLMProvider):
    """Returns queued replies (strings) or raises queued exceptions, and records calls."""

    name = "scripted"
    model = "scripted-test-model"

    def __init__(self, replies=(), available: bool = True):
        self.replies = list(replies)
        self.available = available
        self.calls: list[list[dict]] = []

    def check_status(self) -> ProviderStatus:
        if self.available:
            return ProviderStatus(True, "scripted model ready")
        return ProviderStatus(False, "scripted model is unavailable")

    def chat(self, messages, *, json_mode=False, temperature=None) -> str:
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


@pytest.fixture
def full_project():
    return sample_full()


@pytest.fixture
def brief_project():
    return sample_brief()


def script_reply(project=None) -> dict:
    p = project or sample_full()
    return {
        "logline": p.logline, "synopsis": p.synopsis,
        "characters": [c.model_dump(mode="json") for c in p.characters],
        "scenes": [s.model_dump(mode="json") for s in p.script_scenes],
    }


def scenes_reply(project=None) -> dict:
    p = project or sample_full()
    return {"scenes": [s.model_dump(mode="json") for s in p.scenes]}


def shots_reply(project=None) -> dict:
    p = project or sample_full()
    return {"shots": [s.model_dump(mode="json") for s in p.shots]}
