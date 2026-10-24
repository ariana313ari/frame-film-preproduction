"""UI smoke tests using Streamlit's headless AppTest. Only deterministic paths are exercised,
so neither a network nor a running Ollama is needed."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from schemas import Stage  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def click(at: AppTest, label_part: str) -> None:
    button = next(b for b in at.button if label_part in b.label)
    button.click()
    at.run()


def test_app_renders_and_state_survives_reruns():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Idea & Story", "Script & Scenes", "Shot List",
                                          "Production Plan", "Validation"]

    click(at, "бүрэн demo")
    assert not at.exception
    project = at.session_state.project
    assert len(project.scenes) == 4 and len(project.shots) == 15

    # unrelated reruns must not lose the project
    at.sidebar.text_input(key="w_genre").set_value("Triller").run()
    assert at.session_state.project.genre == "Triller"
    assert len(at.session_state.project.shots) == 15

    click(at, "Шалгалт ажиллуулах")
    assert at.session_state.project.is_fresh(Stage.VALIDATION)
    click(at, "багц бэлтгэх")
    assert not at.exception
    assert "# The Last Bus" in at.session_state.export["markdown"]


def test_flawed_sample_shows_validation_findings():
    at = AppTest.from_file(APP, default_timeout=60).run()
    click(at, "алдаатай жишээ")
    click(at, "Шалгалт ажиллуулах")
    assert not at.exception
    codes = {i.code for i in at.session_state.project.issues}
    assert {"SHOT_SCENE_MISSING", "LOCATION_LIMIT", "CHARACTER_UNKNOWN"} <= codes
    assert any("SHOT_SCENE_MISSING" in m.value for m in at.markdown)


def test_llm_action_without_model_shows_clear_error_and_keeps_state():
    at = AppTest.from_file(APP, default_timeout=60).run()
    click(at, "зөвхөн brief")
    before = at.session_state.project.model_dump_json()
    # Point the provider at a closed port so the test never depends on a local Ollama.
    from llm_provider import LLMConfig, OllamaProvider
    at.session_state["provider"] = OllamaProvider(LLMConfig(base_url="http://127.0.0.1:9", timeout_seconds=2))
    click(at, "Сценар үүсгэх")
    assert not at.exception
    assert "Ollama" in at.session_state.chat[-1]["content"]
    assert at.session_state.chat[-1]["ok"] is False
    assert at.session_state.project.model_dump_json() == before
