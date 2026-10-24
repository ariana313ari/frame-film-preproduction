"""Provider selection and missing-key behaviour for deployment (Streamlit Community Cloud).

Everything here is offline: Ollama is booby-trapped so any attempt to contact it fails the test,
fake keys are made-up strings, and the Gemini SDK client is replaced by a fake.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest
import requests

import llm_provider
from agent import FrameAgent
from llm_provider import (
    SECRET_KEYS,
    GeminiProvider,
    LLMAuthError,
    LLMConfig,
    LLMModelNotFoundError,
    OllamaProvider,
    ProviderStatus,
    apply_secrets,
    create_provider,
)
from sample_project import sample_brief
from schemas import Intent

FAKE_KEY = "AIzaSyDEPLOYFAKEDEPLOYFAKEDEPLOYFAKE012345"
MODEL = "gemini-deploy-test"
REAL_APPLY_SECRETS = apply_secrets  # conftest replaces llm_provider.apply_secrets with a no-op per test


@pytest.fixture
def ollama_is_forbidden(monkeypatch):
    """Any attempt to construct, probe or call Ollama (or to make a requests-based HTTP call) fails."""
    def boom(*args, **kwargs):
        raise AssertionError("Ollama / local HTTP must not be contacted")

    monkeypatch.setattr(OllamaProvider, "__init__", boom)
    monkeypatch.setattr(OllamaProvider, "check_status", boom)
    monkeypatch.setattr(OllamaProvider, "chat", boom)
    monkeypatch.setattr(requests.Session, "request", boom)


# --------------------------------------------------------------------------- #
# Which provider is selected
# --------------------------------------------------------------------------- #
def test_gemini_is_the_default_provider():
    assert LLMConfig().provider == "gemini"
    assert LLMConfig.from_env({}).provider == "gemini"
    assert isinstance(create_provider(LLMConfig.from_env({})), GeminiProvider)


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_settings_count_as_not_set(blank):
    """An empty Streamlit secret / env value must not crash the app or pick a different provider."""
    cfg = LLMConfig.from_env({"FRAME_LLM_PROVIDER": blank, "OLLAMA_BASE_URL": blank, "FRAME_MODEL": blank})
    assert cfg.provider == "gemini"
    assert cfg.base_url == "http://localhost:11434" and cfg.model == "qwen2.5:3b"


def test_ollama_only_when_explicitly_selected():
    for value in ("ollama", "OLLAMA", "  Ollama "):
        provider = create_provider(LLMConfig.from_env({"FRAME_LLM_PROVIDER": value}))
        assert isinstance(provider, OllamaProvider)
    cfg = LLMConfig.from_env({"FRAME_LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "https://ollama.example.org/"})
    assert cfg.base_url == "https://ollama.example.org"          # an explicit endpoint is honoured


def test_unknown_provider_fails_loudly_instead_of_falling_back():
    with pytest.raises(ValueError, match="Unknown FRAME_LLM_PROVIDER 'olama'"):
        create_provider(LLMConfig.from_env({"FRAME_LLM_PROVIDER": "olama"}))


def test_default_selection_never_touches_ollama(ollama_is_forbidden):
    provider = create_provider(LLMConfig.from_env({}))
    assert isinstance(provider, GeminiProvider)
    status = provider.check_status()                                # no key configured -> no network at all
    assert not status.available


# --------------------------------------------------------------------------- #
# Missing API key / model (what a mis-configured deployment looks like)
# --------------------------------------------------------------------------- #
def test_missing_api_key_is_reported_clearly_and_nothing_is_sent(ollama_is_forbidden):
    calls = []
    provider = GeminiProvider(LLMConfig(provider="gemini", gemini_model=MODEL),
                              client_factory=lambda timeout: calls.append(timeout))
    status = provider.check_status()
    assert not status.available
    assert "GEMINI_API_KEY" in status.message and "Secrets" in status.message
    assert "Тохиргоо дутуу" in status.message                        # "configuration incomplete", not a connection error
    with pytest.raises(LLMAuthError, match="GEMINI_API_KEY"):
        provider.chat([{"role": "user", "content": "hi"}])
    assert calls == []                                                # the SDK client was never even created


def test_missing_model_with_key_is_reported(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    provider = GeminiProvider(LLMConfig(provider="gemini"), client_factory=lambda t: None)
    status = provider.check_status()
    assert not status.available and "GEMINI_MODEL" in status.message and FAKE_KEY not in status.message
    with pytest.raises(LLMModelNotFoundError, match="GEMINI_MODEL"):
        provider.chat([{"role": "user", "content": "hi"}])


def test_agent_reports_missing_key_and_leaves_project_untouched(ollama_is_forbidden):
    provider = create_provider(LLMConfig.from_env({}))
    project = sample_brief()
    before = project.model_dump_json()
    turn = FrameAgent(provider).handle(project, "", forced_intent=Intent.BUILD_SCRIPT)
    assert not turn.ok and "GEMINI_API_KEY" in turn.reply and turn.steps == []
    assert project.model_dump_json() == before


# --------------------------------------------------------------------------- #
# Streamlit secrets bridge
# --------------------------------------------------------------------------- #
def register_cleanup(monkeypatch, *names):
    """apply_secrets writes os.environ directly: pre-register so monkeypatch removes it afterwards."""
    for name in names:
        monkeypatch.setenv(name, "")


def test_apply_secrets_copies_only_allowlisted_plain_strings(monkeypatch):
    register_cleanup(monkeypatch, *SECRET_KEYS)
    applied = apply_secrets({
        "GEMINI_API_KEY": f"  {FAKE_KEY} ", "GEMINI_MODEL": MODEL, "FRAME_LLM_PROVIDER": "gemini",
        "SOMETHING_ELSE": "ignored",                                  # not allowlisted
        "OLLAMA_BASE_URL": {"nested": "table"},                       # nested tables are not used
        "FRAME_MODEL": "   ",                                         # blank
        "FRAME_NUM_CTX": 8192,                                        # non-string
    })
    assert sorted(applied) == ["FRAME_LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL"]
    assert os.environ["GEMINI_API_KEY"] == FAKE_KEY and os.environ["GEMINI_MODEL"] == MODEL
    assert "SOMETHING_ELSE" not in os.environ
    assert not os.environ.get("OLLAMA_BASE_URL") and not os.environ.get("FRAME_MODEL")
    assert FAKE_KEY not in " ".join(applied)                          # only names are ever returned


def test_apply_secrets_never_overrides_a_set_environment_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "from-environment")
    monkeypatch.setenv("GEMINI_API_KEY", "")                          # blank in env -> filled from secrets
    applied = apply_secrets({"GEMINI_MODEL": "from-secrets", "GEMINI_API_KEY": FAKE_KEY})
    assert os.environ["GEMINI_MODEL"] == "from-environment" and applied == ["GEMINI_API_KEY"]
    assert os.environ["GEMINI_API_KEY"] == FAKE_KEY


def test_apply_secrets_tolerates_missing_secrets():
    class NoSecretsFile(Mapping):
        def __getitem__(self, key):
            raise FileNotFoundError("No secrets found")

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

        def get(self, key, default=None):
            raise FileNotFoundError("No secrets found")                # what Streamlit does without a secrets file

    assert apply_secrets(NoSecretsFile()) == []
    assert apply_secrets({}) == []


def test_secrets_and_config_end_to_end(monkeypatch):
    register_cleanup(monkeypatch, *SECRET_KEYS)
    apply_secrets({"GEMINI_API_KEY": FAKE_KEY, "GEMINI_MODEL": MODEL})
    cfg = LLMConfig.from_env()
    assert cfg.provider == "gemini" and cfg.gemini_model == MODEL
    assert FAKE_KEY not in repr(cfg)


def test_secrets_are_not_committed_or_templated_with_real_values():
    root = Path(__file__).resolve().parent.parent
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".streamlit/secrets.toml" in gitignore and ".env" in gitignore
    template = (root / ".streamlit" / "secrets.toml.example").read_text(encoding="utf-8")
    assert "AIza" not in template                                     # placeholder only
    assert 'FRAME_LLM_PROVIDER = "gemini"' in template


# --------------------------------------------------------------------------- #
# The Streamlit app itself (headless): no key, secrets-provided key, explicit Ollama
# --------------------------------------------------------------------------- #
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app(monkeypatch, secrets=None):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(llm_provider, "apply_secrets", REAL_APPLY_SECRETS)   # use the real bridge in the app
    register_cleanup(monkeypatch, *SECRET_KEYS)
    at = AppTest.from_file(APP, default_timeout=60)
    for key, value in (secrets or {}).items():
        at.secrets[key] = value
    return at.run()


def test_app_without_any_config_defaults_to_gemini_and_never_contacts_ollama(monkeypatch, ollama_is_forbidden):
    at = run_app(monkeypatch)
    assert not at.exception
    assert at.session_state.llm_config.provider == "gemini"
    assert isinstance(at.session_state.provider, GeminiProvider)
    status = at.session_state.llm_status
    assert not status.available and "GEMINI_API_KEY" in status.message
    assert any("Demo" in w.value for w in at.warning)                  # clear banner, rule-based tools still usable
    assert [t.key for t in at.sidebar.text_input if t.key in ("w_base_url", "w_model")] == []   # no Ollama fields
    next(b for b in at.button if "зөвхөн brief" in b.label).click()   # a project with premise + genre
    at.run()
    before = at.session_state.project.model_dump_json()
    next(b for b in at.button if "Сценар үүсгэх" in b.label).click()
    at.run()
    assert not at.exception and "GEMINI_API_KEY" in at.session_state.chat[-1]["content"]
    assert at.session_state.chat[-1]["ok"] is False
    assert at.session_state.project.model_dump_json() == before         # nothing was generated or changed


def test_app_reads_key_and_model_from_streamlit_secrets(monkeypatch, ollama_is_forbidden):
    class FakeClient:
        models = type("M", (), {"get": staticmethod(lambda **kw: object())})()

        def close(self):
            pass

    monkeypatch.setitem(llm_provider.PROVIDERS, "gemini",
                        lambda cfg: GeminiProvider(cfg, client_factory=lambda timeout: FakeClient()))
    at = run_app(monkeypatch, {"GEMINI_API_KEY": FAKE_KEY, "GEMINI_MODEL": MODEL})
    assert not at.exception
    assert at.session_state.llm_config.provider == "gemini" and at.session_state.provider.model == MODEL
    assert at.session_state.llm_status.available
    dump = repr(at.session_state)
    assert FAKE_KEY not in dump                                        # the key never enters session state
    shown = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert FAKE_KEY not in shown and f"gemini · {MODEL}" in shown


def test_app_uses_ollama_only_when_selected_in_secrets(monkeypatch):
    class StubOllama(OllamaProvider):
        def check_status(self):
            return ProviderStatus(True, "stub ollama ready")

    monkeypatch.setitem(llm_provider.PROVIDERS, "ollama", StubOllama)
    at = run_app(monkeypatch, {"FRAME_LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "https://ollama.example.org"})
    assert not at.exception
    assert at.session_state.llm_config.provider == "ollama"
    assert at.session_state.llm_config.base_url == "https://ollama.example.org"
    assert isinstance(at.session_state.provider, OllamaProvider)
    assert [t.key for t in at.sidebar.text_input if t.key == "w_base_url"] == ["w_base_url"]   # Ollama settings shown
