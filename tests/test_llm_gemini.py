"""Gemini provider tests. Fully offline: the SDK client is replaced by a fake and the API key
below is a made-up string. No real network request is made and no real key is needed."""
from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import ScriptedLLM

import llm_provider
from agent import FrameAgent
from llm_provider import (
    GeminiProvider,
    InvalidModelOutputError,
    LLMAuthError,
    LLMConfig,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    OllamaProvider,
    create_provider,
    generate_validated,
    load_env_file,
    to_gemini_contents,
)
from prompts import general_answer_messages
from sample_project import sample_brief
from schemas import Intent, SceneBreakdownOutput

FAKE_KEY = "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKE0123456789"
MODEL = "gemini-test-model"
MSGS = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "Plan my film"},
]


class FakeAPIError(Exception):
    """Shaped like google.genai.errors.APIError (code / status / message)."""

    def __init__(self, code, message="", status=""):
        super().__init__(f"{code} {status}. {message}")
        self.code, self.message, self.status = code, message, status


class FakeClient:
    def __init__(self, *, text='{"ok": true}', response=None, get_error=None, gen_error=None):
        self.response = response if response is not None else SimpleNamespace(
            text=text, prompt_feedback=None, candidates=[])
        self.get_error, self.gen_error = get_error, gen_error
        self.get_calls: list[dict] = []
        self.gen_calls: list[dict] = []
        self.closed = False
        self.models = SimpleNamespace(get=self._get, generate_content=self._generate)

    def _get(self, **kwargs):
        self.get_calls.append(kwargs)
        if self.get_error:
            raise self.get_error
        return SimpleNamespace(name=kwargs["model"])

    def _generate(self, **kwargs):
        self.gen_calls.append(kwargs)
        if self.gen_error:
            raise self.gen_error
        return self.response

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)


def provider(client=None, **config):
    client = client or FakeClient()
    cfg = LLMConfig(provider="gemini", gemini_model=config.pop("model", MODEL), **config)
    prov = GeminiProvider(cfg, client_factory=lambda timeout: client)
    return prov, client


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #
def test_provider_selection_from_env():
    gem = create_provider(LLMConfig.from_env({"FRAME_LLM_PROVIDER": "gemini", "GEMINI_MODEL": MODEL}))
    assert isinstance(gem, GeminiProvider) and gem.model == MODEL
    oll = create_provider(LLMConfig.from_env({"FRAME_LLM_PROVIDER": "ollama", "FRAME_MODEL": "qwen2.5:3b"}))
    assert isinstance(oll, OllamaProvider) and oll.model == "qwen2.5:3b"
    assert isinstance(create_provider(LLMConfig.from_env({})), GeminiProvider)   # Gemini is the default
    assert {"ollama", "gemini"} <= set(llm_provider.PROVIDERS)


def test_unknown_provider_is_an_error_not_a_fallback():
    with pytest.raises(ValueError, match="Unknown FRAME_LLM_PROVIDER 'geminii'"):
        create_provider(LLMConfig.from_env({"FRAME_LLM_PROVIDER": "geminii"}))


def test_frame_model_keeps_its_meaning_and_gemini_uses_its_own_setting():
    cfg = LLMConfig.from_env({"FRAME_LLM_PROVIDER": "gemini", "FRAME_MODEL": "qwen2.5:7b", "GEMINI_MODEL": MODEL})
    assert cfg.model == "qwen2.5:7b" and cfg.gemini_model == MODEL
    assert LLMConfig.from_env({}).gemini_model == "" and LLMConfig.from_env({}).gemini_temperature is None
    assert LLMConfig.from_env({"GEMINI_TEMPERATURE": "0.7"}).gemini_temperature == 0.7
    assert LLMConfig.from_env({"GEMINI_TEMPERATURE": "abc"}).gemini_temperature is None


# --------------------------------------------------------------------------- #
# Configuration problems vs confirmed connection
# --------------------------------------------------------------------------- #
def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    prov, client = provider()
    status = prov.check_status()
    assert not status.available and "GEMINI_API_KEY" in status.message and "Тохиргоо дутуу" in status.message
    with pytest.raises(LLMAuthError, match="GEMINI_API_KEY"):
        prov.chat(MSGS)
    assert client.get_calls == [] and client.gen_calls == []        # nothing was sent anywhere


def test_blank_api_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert not provider()[0].check_status().available


def test_missing_model():
    prov, client = provider(model="")
    status = prov.check_status()
    assert not status.available and "GEMINI_MODEL" in status.message
    with pytest.raises(LLMModelNotFoundError, match="GEMINI_MODEL"):
        prov.chat(MSGS)
    assert client.get_calls == [] and client.gen_calls == []


def test_status_confirms_connection_without_generating():
    prov, client = provider()
    status = prov.check_status()
    assert status.available and MODEL in status.message and "баталгаажсан" in status.message
    assert client.get_calls == [{"model": MODEL}] and client.gen_calls == []
    assert client.closed
    prov.check_status()                                              # confirmed status is briefly reused
    assert len(client.get_calls) == 1


def test_failed_status_is_not_cached():
    client = FakeClient(get_error=FakeAPIError(503, "overloaded", "UNAVAILABLE"))
    prov, _ = provider(client)
    assert not prov.check_status().available
    client.get_error = None
    assert prov.check_status().available and len(client.get_calls) == 2


@pytest.mark.parametrize("error, fragment", [
    (FakeAPIError(404, "model not found", "NOT_FOUND"), "олдсонгүй"),
    (FakeAPIError(403, "denied", "PERMISSION_DENIED"), "түлхүүр"),
    (FakeAPIError(429, "quota", "RESOURCE_EXHAUSTED"), "хязгаар"),
    (httpx.ConnectError("boom"), "холбогдож"),
])
def test_status_reports_api_problems(error, fragment):
    status = provider(FakeClient(get_error=error))[0].check_status()
    assert not status.available and fragment in status.message


def test_status_never_raises_even_for_unexpected_errors():
    status = provider(FakeClient(get_error=RuntimeError("weird")))[0].check_status()
    assert not status.available and "RuntimeError" in status.message


# --------------------------------------------------------------------------- #
# Message conversion
# --------------------------------------------------------------------------- #
def test_message_conversion_maps_roles_and_keeps_order():
    system, contents = to_gemini_contents(MSGS)
    assert system == "Be brief."
    assert contents == [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi!"}]},
        {"role": "user", "parts": [{"text": "Plan my film"}]},
    ]


def test_conversion_merges_same_role_turns_and_joins_system_messages():
    system, contents = to_gemini_contents([
        {"role": "system", "content": "A"}, {"role": "system", "content": "B"},
        {"role": "user", "content": "one"}, {"role": "user", "content": "two"},
        {"role": "user", "content": "   "},                           # blank: cannot be sent
        {"role": "assistant", "content": "three"},
    ])
    assert system == "A\n\nB"
    assert contents == [{"role": "user", "parts": [{"text": "one"}, {"text": "two"}]},
                        {"role": "model", "parts": [{"text": "three"}]}]


def test_conversion_without_system_message_and_invalid_input():
    assert to_gemini_contents([{"role": "user", "content": "hi"}])[0] is None
    with pytest.raises(ValueError, match="role"):
        to_gemini_contents([{"role": "tool", "content": "x"}])
    with pytest.raises(ValueError, match="string"):
        to_gemini_contents([{"role": "user", "content": None}])
    with pytest.raises(ValueError, match="non-empty"):
        to_gemini_contents([{"role": "system", "content": "only system"}])


def test_chat_sends_converted_messages_and_returns_text():
    prov, client = provider(FakeClient(text="Сайн байна уу"))
    assert prov.chat(MSGS) == "Сайн байна уу"
    call = client.gen_calls[0]
    assert call["model"] == MODEL
    assert [c["role"] for c in call["contents"]] == ["user", "model", "user"]
    assert call["config"] == {"system_instruction": "Be brief."}
    assert client.closed


# --------------------------------------------------------------------------- #
# JSON mode and temperature
# --------------------------------------------------------------------------- #
def test_json_mode_requests_json_mime_type():
    prov, client = provider()
    prov.chat(MSGS, json_mode=True)
    assert client.gen_calls[0]["config"]["response_mime_type"] == "application/json"
    prov.chat(MSGS)
    assert "response_mime_type" not in client.gen_calls[1]["config"]


def test_temperature_is_not_sent_unless_configured():
    prov, client = provider()
    prov.chat(MSGS, temperature=0.1)                                  # per-call value is ignored on purpose
    assert "temperature" not in client.gen_calls[0]["config"]
    prov, client = provider(gemini_temperature=0.7)
    prov.chat(MSGS, temperature=0.1)
    assert client.gen_calls[0]["config"]["temperature"] == 0.7


def test_timeout_comes_from_config():
    seen = []
    prov = GeminiProvider(LLMConfig(provider="gemini", gemini_model=MODEL, timeout_seconds=42),
                          client_factory=lambda t: seen.append(t) or FakeClient())
    prov.chat(MSGS)
    assert seen == [42]


def test_generate_validated_works_with_gemini_including_the_repair_turn():
    good = {"scenes": [{"number": 1, "location": "Bus stop", "estimated_duration_sec": 30}]}
    client = FakeClient()
    replies = iter(["not json at all", json.dumps(good)])
    client.response = None
    client._generate = lambda **kw: (client.gen_calls.append(kw),
                                     SimpleNamespace(text=next(replies), prompt_feedback=None, candidates=[]))[1]
    client.models = SimpleNamespace(get=client._get, generate_content=client._generate)
    prov, _ = provider(client)
    out = generate_validated(prov, [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
                             SceneBreakdownOutput)
    assert out.scenes[0].location == "Bus stop"
    repair_roles = [c["role"] for c in client.gen_calls[1]["contents"]]
    assert repair_roles == ["user", "model", "user"]                  # original ask, bad answer, repair request
    assert "rejected" in client.gen_calls[1]["contents"][2]["parts"][0]["text"]


def test_invalid_json_twice_raises_instead_of_fabricating():
    prov, client = provider(FakeClient(text="nope"))
    with pytest.raises(InvalidModelOutputError):
        generate_validated(prov, [{"role": "user", "content": "Q"}], SceneBreakdownOutput)
    assert len(client.gen_calls) == 2


# --------------------------------------------------------------------------- #
# Empty / blocked responses
# --------------------------------------------------------------------------- #
def test_blocked_prompt_is_reported():
    blocked = SimpleNamespace(text=None, candidates=[],
                              prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name="SAFETY")))
    with pytest.raises(LLMResponseError, match="SAFETY"):
        provider(FakeClient(response=blocked))[0].chat(MSGS)


def test_empty_response_is_an_error_not_an_empty_success():
    empty = SimpleNamespace(text="  ", prompt_feedback=None,
                            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))])
    with pytest.raises(LLMResponseError, match="MAX_TOKENS"):
        provider(FakeClient(response=empty))[0].chat(MSGS)


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("error, expected", [
    (FakeAPIError(401, "unauthenticated", "UNAUTHENTICATED"), LLMAuthError),
    (FakeAPIError(403, "denied", "PERMISSION_DENIED"), LLMAuthError),
    (FakeAPIError(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"), LLMAuthError),
    (FakeAPIError(400, "unsupported parameter", "INVALID_ARGUMENT"), LLMResponseError),
    (FakeAPIError(404, "not found", "NOT_FOUND"), LLMModelNotFoundError),
    (FakeAPIError(429, "quota", "RESOURCE_EXHAUSTED"), LLMRateLimitError),
    (FakeAPIError(504, "deadline", "DEADLINE_EXCEEDED"), LLMTimeoutError),
    (FakeAPIError(500, "internal", "INTERNAL"), LLMResponseError),
    (FakeAPIError(503, "overloaded", "UNAVAILABLE"), LLMResponseError),
    (FakeAPIError(418, "teapot", "TEAPOT"), LLMResponseError),
    (httpx.ReadTimeout("slow"), LLMTimeoutError),
    (httpx.ConnectTimeout("slow"), LLMTimeoutError),
    (httpx.ConnectError("down"), LLMConnectionError),
    (ConnectionResetError("reset"), LLMConnectionError),
    (RuntimeError("unexpected"), LLMResponseError),
])
def test_error_mapping(error, expected):
    prov, client = provider(FakeClient(gen_error=error))
    with pytest.raises(expected) as caught:
        prov.chat(MSGS)
    assert type(caught.value) is expected
    assert client.closed                                              # client released even on failure


def test_errors_never_contain_the_api_key():
    leaky = [
        FakeAPIError(400, f"bad request for key {FAKE_KEY}", "INVALID_ARGUMENT"),
        FakeAPIError(418, f"teapot {FAKE_KEY}", "TEAPOT"),
        FakeAPIError(500, f"internal {FAKE_KEY}", "INTERNAL"),
        RuntimeError(f"url?key={FAKE_KEY}"),
        ValueError("AIzaSyANOTHERKEYSHAPEDSTRING_1234567890"),
    ]
    for error in leaky:
        with pytest.raises(LLMError) as caught:
            provider(FakeClient(gen_error=error))[0].chat(MSGS)
        assert FAKE_KEY not in str(caught.value) and "AIzaSyANOTHER" not in str(caught.value)
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
        status = provider(FakeClient(get_error=error))[0].check_status()
        assert FAKE_KEY not in status.message


def test_real_sdk_error_classes_are_mapped():
    errors = pytest.importorskip("google.genai.errors")
    body = {"error": {"code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    with pytest.raises(LLMRateLimitError):
        provider(FakeClient(gen_error=errors.ClientError(429, body)))[0].chat(MSGS)
    body = {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}
    with pytest.raises(LLMAuthError):
        provider(FakeClient(gen_error=errors.ClientError(403, body)))[0].chat(MSGS)
    body = {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}}
    with pytest.raises(LLMResponseError):
        provider(FakeClient(gen_error=errors.ServerError(503, body)))[0].chat(MSGS)


def test_real_sdk_client_accepts_our_requests_and_errors_are_mapped():
    """Uses the real google-genai client, pointed at a closed loopback port (no internet)."""
    genai = pytest.importorskip("google.genai")

    def real_client(timeout_seconds):
        return genai.Client(api_key=FAKE_KEY, http_options={"base_url": "http://127.0.0.1:9", "timeout": 1500})

    prov = GeminiProvider(LLMConfig(provider="gemini", gemini_model=MODEL, timeout_seconds=2),
                          client_factory=real_client)
    # The SDK validated our dict payload (system_instruction, JSON mime type, roles) before connecting.
    with pytest.raises((LLMTimeoutError, LLMConnectionError)) as caught:
        prov.chat(MSGS, json_mode=True)
    assert FAKE_KEY not in str(caught.value)
    status = prov.check_status()
    assert not status.available and FAKE_KEY not in status.message


def test_missing_sdk_gives_install_hint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google" and fromlist and "genai" in fromlist:
            raise ImportError("no google-genai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    prov = GeminiProvider(LLMConfig(provider="gemini", gemini_model=MODEL))
    status = prov.check_status()
    assert not status.available and "pip install" in status.message
    with pytest.raises(LLMError, match="google-genai"):
        prov.chat(MSGS)


# --------------------------------------------------------------------------- #
# The key stays out of objects that live in Streamlit session state
# --------------------------------------------------------------------------- #
def test_api_key_is_not_stored_on_provider_or_config():
    prov, _ = provider()
    prov.check_status()
    prov.chat(MSGS)
    assert FAKE_KEY not in repr(vars(prov)) and FAKE_KEY not in repr(prov.config)
    cfg = LLMConfig.from_env({"FRAME_LLM_PROVIDER": "gemini", "GEMINI_API_KEY": FAKE_KEY})
    assert FAKE_KEY not in repr(cfg)


# --------------------------------------------------------------------------- #
# Agent integration: no silent fallback
# --------------------------------------------------------------------------- #
def test_agent_reports_gemini_configuration_problem_and_keeps_state(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    project = sample_brief()
    before = project.model_dump_json()
    turn = FrameAgent(provider()[0]).handle(project, "", forced_intent=Intent.BUILD_SCRIPT)
    assert not turn.ok and "GEMINI_API_KEY" in turn.reply and turn.steps == []
    assert project.model_dump_json() == before


def test_agent_runs_a_tool_through_gemini(brief_project):
    from conftest import script_reply
    client = FakeClient(text=json.dumps(script_reply(), ensure_ascii=False))
    turn = FrameAgent(provider(client)[0]).handle(brief_project, "", forced_intent=Intent.BUILD_SCRIPT)
    assert turn.ok and len(brief_project.script_scenes) == 4
    assert brief_project.provenance[next(iter(brief_project.provenance))] == f"model:{MODEL}"
    assert client.gen_calls[0]["config"]["response_mime_type"] == "application/json"


# --------------------------------------------------------------------------- #
# .env loading and prompt history
# --------------------------------------------------------------------------- #
def test_env_file_loading(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_bytes(
        "﻿FRAME_T_FIRST=one\n# comment\nexport FRAME_T_EXPORTED='two'\nFRAME_T_EMPTY_IN_ENV=three\n"
        "FRAME_T_KEEP=from_file\nFRAME_T_BLANK=\nnot a pair\n".encode("utf-8"))
    for name in ("FRAME_T_FIRST", "FRAME_T_EXPORTED", "FRAME_T_BLANK"):
        monkeypatch.setenv(name, "")                                  # registers cleanup; empty == unset
    monkeypatch.setenv("FRAME_T_EMPTY_IN_ENV", "")
    monkeypatch.setenv("FRAME_T_KEEP", "from_environment")
    load_env_file(env_file)
    import os
    assert os.environ["FRAME_T_FIRST"] == "one"                       # BOM did not corrupt the first key
    assert os.environ["FRAME_T_EXPORTED"] == "two"
    assert os.environ["FRAME_T_EMPTY_IN_ENV"] == "three"              # empty variable is filled from the file
    assert os.environ["FRAME_T_KEEP"] == "from_environment"           # a real value always wins
    assert os.environ["FRAME_T_BLANK"] == ""


def test_missing_env_file_is_fine(tmp_path):
    load_env_file(tmp_path / "does-not-exist.env")


def test_general_answer_history_never_starts_with_an_assistant_turn():
    history = [{"role": "assistant", "content": "Greeting"}, {"role": "user", "content": "q1"},
               {"role": "assistant", "content": "a1"}]
    messages = general_answer_messages(project_summary="p", history=history, user_message="q2", language="mn")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    system, contents = to_gemini_contents(messages)
    assert contents[0]["role"] == "user" and system


# --------------------------------------------------------------------------- #
# UI: Gemini selected, no network, no secret in session state
# --------------------------------------------------------------------------- #
def test_ui_with_gemini_provider(monkeypatch):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FRAME_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", MODEL)
    monkeypatch.setitem(
        llm_provider.PROVIDERS, "gemini",
        lambda cfg: GeminiProvider(cfg, client_factory=lambda timeout: FakeClient(text="Сайн уу")))
    app = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app, default_timeout=60).run()

    assert not at.exception
    assert at.session_state.llm_status.available
    assert at.session_state.provider.model == MODEL
    assert [t.key for t in at.sidebar.text_input if t.key in ("w_base_url", "w_model")] == []   # Ollama inputs hidden
    shown = " ".join(m.value for m in at.sidebar.markdown) + " ".join(c.value for c in at.sidebar.caption)
    assert f"gemini · {MODEL}" in shown and "Google" in shown       # chip + free-tier privacy notice
    assert FAKE_KEY not in shown

    state_dump = repr(at.session_state)
    assert "llm_config" in state_dump and "provider" in state_dump        # the dump really covers those objects
    assert FAKE_KEY not in state_dump

    next(b for b in at.button if "Холболтыг дахин шалгах" in b.label).click()
    at.run()
    assert not at.exception and at.session_state.llm_status.available


def test_ui_with_gemini_but_missing_key_shows_configuration_problem(monkeypatch):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("FRAME_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", MODEL)
    app = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app, default_timeout=60).run()
    assert not at.exception
    status = at.session_state.llm_status
    assert not status.available and "GEMINI_API_KEY" in status.message
    assert any("Demo" in w.value for w in at.warning)               # clear banner, no silent switch to Ollama
    assert at.session_state.llm_config.provider == "gemini"


def test_scripted_llm_helper_is_unaffected():
    assert ScriptedLLM(["x"]).chat([{"role": "user", "content": "q"}]) == "x"
