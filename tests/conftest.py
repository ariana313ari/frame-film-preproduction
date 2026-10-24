"""Test helpers. The scripted LLM below exists ONLY in tests: the application never
substitutes canned output for a real model response."""
from __future__ import annotations

import json

import pytest

from llm_provider import LLMProvider, ProviderStatus
from sample_project import sample_brief, sample_full


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
