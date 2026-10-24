"""LLM provider adapter.

`LLMProvider` is the only thing the agent and tools know about. Google Gemini is the
default provider; local Ollama is optional and used ONLY when it is explicitly selected
(`FRAME_LLM_PROVIDER=ollama`), so a hosted deployment never probes a local Ollama server.
Another API can be supported by subclassing `LLMProvider` and registering it in
`PROVIDERS` (see README, "Adding a provider"). The active provider is never changed
silently: a configuration or API failure is raised, not worked around.

Nothing in this module fabricates model output: if the model cannot be reached,
a typed `LLMError` is raised and surfaced to the user.
"""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, TypeVar

import requests
from pydantic import BaseModel, ValidationError

import prompts


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Base class for all provider problems."""


class LLMConnectionError(LLMError):
    """The server could not be reached."""


class LLMTimeoutError(LLMError):
    """The server did not answer in time."""


class LLMModelNotFoundError(LLMError):
    """The server is up but the configured model is not installed."""


class LLMResponseError(LLMError):
    """The server answered with an error or an unusable payload."""


class LLMAuthError(LLMError):
    """Missing, invalid or unauthorised API key (never includes the key itself)."""


class LLMRateLimitError(LLMError):
    """The provider's rate limit or quota was exceeded."""


class InvalidModelOutputError(LLMError):
    """The model's output stayed invalid after the allowed repair attempt."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_env_file(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines, optional `export ` prefix).

    A non-empty variable that is already in the environment wins; an empty or
    missing one is filled from the file. `utf-8-sig` tolerates the BOM that
    Windows editors add (it would otherwise corrupt the first key).
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


# Settings that may be supplied through Streamlit secrets (Community Cloud "Secrets" or a local
# .streamlit/secrets.toml). Only these names are ever copied into the environment.
SECRET_KEYS = (
    "FRAME_LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_TEMPERATURE",
    "OLLAMA_BASE_URL", "FRAME_MODEL", "FRAME_TIMEOUT_SECONDS", "FRAME_TEMPERATURE", "FRAME_NUM_CTX",
)


def apply_secrets(secrets: Mapping[str, object]) -> list[str]:
    """Copy allowlisted top-level secrets into `os.environ` and return the NAMES that were set.

    Like `load_env_file`, a non-empty variable that is already set wins, so a real environment
    variable can always override a secret. Only plain strings are used (nested tables are
    ignored) and values are never returned or logged. Raises nothing if secrets are unavailable.
    """
    applied: list[str] = []
    for name in SECRET_KEYS:
        try:
            value = secrets.get(name)
        except (FileNotFoundError, KeyError):  # Streamlit raises FileNotFoundError when no secrets exist
            return applied
        if isinstance(value, str) and value.strip() and not os.environ.get(name):
            os.environ[name] = value.strip()
            applied.append(name)
    return applied


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "gemini"  # Ollama only when FRAME_LLM_PROVIDER=ollama is set explicitly
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    timeout_seconds: float = 300.0  # CPU-only inference can be slow
    temperature: float = 0.4
    num_ctx: int = 8192
    # Gemini (non-secret). The API key is deliberately NOT part of the config: it is
    # read from GEMINI_API_KEY at request time so it never sits in session state.
    gemini_model: str = ""
    gemini_temperature: float | None = None  # None = use the model's default

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LLMConfig":
        env = os.environ if env is None else env
        default = cls()

        def number(key: str, fallback, cast):
            try:
                return cast(env[key]) if key in env else fallback
            except ValueError:
                return fallback

        def text(key: str, fallback: str) -> str:  # a blank value counts as "not set"
            return (env.get(key) or "").strip() or fallback

        return cls(
            provider=text("FRAME_LLM_PROVIDER", default.provider).lower(),
            base_url=text("OLLAMA_BASE_URL", default.base_url).rstrip("/"),
            model=text("FRAME_MODEL", default.model),
            timeout_seconds=number("FRAME_TIMEOUT_SECONDS", default.timeout_seconds, float),
            temperature=number("FRAME_TEMPERATURE", default.temperature, float),
            num_ctx=number("FRAME_NUM_CTX", default.num_ctx, int),
            gemini_model=env.get("GEMINI_MODEL", "").strip(),
            gemini_temperature=number("GEMINI_TEMPERATURE", None, float),
        )


@dataclass(frozen=True)
class ProviderStatus:
    available: bool          # server reachable AND model installed
    message: str
    models: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def check_status(self) -> ProviderStatus:
        """Cheap reachability check; must not raise."""

    @abstractmethod
    def chat(self, messages: list[dict], *, json_mode: bool = False,
             temperature: float | None = None) -> str:
        """Send chat messages ([{'role','content'}]) and return the reply text.

        Raises an `LLMError` subclass on any failure.
        """


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.model = self.config.model
        self._session = requests.Session()

    # -- status ----------------------------------------------------------- #
    def check_status(self) -> ProviderStatus:
        url = f"{self.config.base_url}/api/tags"
        try:
            resp = self._session.get(url, timeout=3)
            resp.raise_for_status()
            models = tuple(m.get("name", "") for m in resp.json().get("models", []))
        except requests.RequestException as exc:
            return ProviderStatus(
                False,
                f"Ollama ({self.config.base_url}) руу холбогдож чадсангүй. "
                f"`ollama serve` ажиллаж байгаа эсэхийг шалгана уу. ({type(exc).__name__})",
            )
        except ValueError:
            return ProviderStatus(False, "Ollama-гийн хариу уншигдсангүй (JSON биш).")

        if not self._model_installed(models):
            return ProviderStatus(
                False,
                f"Ollama ажиллаж байна, гэвч '{self.model}' загвар суугаагүй байна. "
                f"`ollama pull {self.model}` ажиллуулна уу.",
                models,
            )
        return ProviderStatus(True, f"Ollama · {self.model} бэлэн", models)

    def _model_installed(self, models: tuple[str, ...]) -> bool:
        wanted = self.model
        candidates = {wanted, wanted if ":" in wanted else f"{wanted}:latest"}
        return any(m in candidates for m in models)

    # -- chat ---------------------------------------------------------------- #
    def chat(self, messages, *, json_mode=False, temperature=None) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature if temperature is None else temperature,
                "num_ctx": self.config.num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"

        url = f"{self.config.base_url}/api/chat"
        try:
            resp = self._session.post(url, json=payload, timeout=(5, self.config.timeout_seconds))
        except requests.ConnectionError as exc:
            raise LLMConnectionError(
                f"Ollama ({self.config.base_url}) руу холбогдож чадсангүй. "
                "Ollama суулгасан эсэх, `ollama serve` ажиллаж байгаа эсэхийг шалгана уу."
            ) from exc
        except requests.Timeout as exc:
            raise LLMTimeoutError(
                f"Загвар {self.config.timeout_seconds:.0f} секундэд хариулсангүй. "
                "FRAME_TIMEOUT_SECONDS-ийг нэмэх эсвэл жижиг загвар ашиглана уу."
            ) from exc
        except requests.RequestException as exc:
            raise LLMResponseError(f"Ollama хүсэлт амжилтгүй: {exc}") from exc

        if resp.status_code == 404:
            raise LLMModelNotFoundError(
                f"'{self.model}' загвар олдсонгүй. `ollama pull {self.model}` ажиллуулах "
                "эсвэл FRAME_MODEL-ийг өөр суулгасан загвар болгож тохируулна уу."
            )
        if resp.status_code >= 400:
            detail = resp.text[:300]
            try:
                detail = resp.json().get("error", detail)
            except ValueError:
                pass
            raise LLMResponseError(f"Ollama алдаа {resp.status_code}: {detail}")

        try:
            content = resp.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMResponseError("Ollama-гийн хариу хүлээгдсэн бүтэцгүй байна.") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("Загвар хоосон хариу буцаалаа.")
        return content


# --------------------------------------------------------------------------- #
# Gemini (Google Gen AI SDK: `google-genai`, `models.generate_content`)
# --------------------------------------------------------------------------- #
GEMINI_KEY_ENV = "GEMINI_API_KEY"
_API_KEY_SHAPE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")  # Google API key format, scrubbed from messages
_GEMINI_STATUS_TIMEOUT = 8.0


def to_gemini_contents(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert FRAME chat messages to (system_instruction, Gemini `contents`).

    * `system` messages become the system instruction (joined in order).
    * `user` -> role "user", `assistant` -> role "model".
    * Order is preserved; consecutive turns of the same role are merged into one
      content with several parts (Gemini expects alternating turns).
    * Whitespace-only messages cannot be sent and are skipped.
    """
    system_parts: list[str] = []
    contents: list[dict] = []
    for message in messages:
        role, text = message.get("role"), message.get("content")
        if not isinstance(text, str):
            raise ValueError("Every chat message needs a string 'content'.")
        if not text.strip():
            continue
        if role == "system":
            system_parts.append(text)
            continue
        gemini_role = {"user": "user", "assistant": "model"}.get(role)
        if gemini_role is None:
            raise ValueError(f"Unsupported message role: {role!r}")
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].append({"text": text})
        else:
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
    if not contents:
        raise ValueError("At least one non-empty user message is required.")
    return ("\n\n".join(system_parts) or None), contents


class GeminiProvider(LLMProvider):
    """Google Gemini through the official `google-genai` SDK.

    * Needs `GEMINI_API_KEY` and `GEMINI_MODEL` (see `.env.example`).
    * The key is read from the process environment for every request and the SDK
      client is short-lived, so no secret is stored on this object, in `LLMConfig`
      or in Streamlit session state. Error text is scrubbed of the key.
    * Temperature: Google recommends the default for Gemini 3 models (lower values
      can cause looping), so none is sent unless `GEMINI_TEMPERATURE` is set. A
      per-call `temperature` argument is therefore ignored by this provider.
    * `client_factory(timeout_seconds)` exists so tests can inject a fake SDK client.
    """

    name = "gemini"
    _STATUS_OK_TTL = 60.0  # a confirmed status is reused briefly (the agent checks once per turn)

    def __init__(self, config: LLMConfig | None = None, *,
                 client_factory: Callable[[float], object] | None = None):
        self.config = config or LLMConfig(provider="gemini")
        self.model = self.config.gemini_model
        self._client_factory = client_factory
        self._status_ok_until = 0.0

    # -- configuration -------------------------------------------------------- #
    @staticmethod
    def _api_key() -> str:
        return os.environ.get(GEMINI_KEY_ENV, "").strip()

    def _config_error(self) -> LLMError | None:
        if not self._api_key():
            return LLMAuthError(
                f"{GEMINI_KEY_ENV} тохируулагдаагүй байна. Streamlit Cloud дээр Secrets-д, локал дээр `.env` "
                "файлд эсвэл орчны хувьсагчид тохируулна уу (түлхүүрийг код эсвэл UI-д бүү оруулаарай).")
        if not self.model:
            return LLMModelNotFoundError(
                "GEMINI_MODEL тохируулагдаагүй байна. Streamlit Secrets эсвэл `.env`-д Gemini загварын нэрийг зааж өгнө үү "
                "(боломжит загваруудыг https://ai.google.dev/gemini-api/docs/models хаягаас үзнэ үү).")
        return None

    def _scrub(self, text: object) -> str:
        text = str(text)
        key = self._api_key()
        if key:
            text = text.replace(key, "[REDACTED]")
        return _API_KEY_SHAPE.sub("[REDACTED]", text)[:300]

    def _new_client(self, timeout_seconds: float):
        if self._client_factory is not None:
            return self._client_factory(timeout_seconds)
        try:
            from google import genai
        except ImportError:
            raise LLMError(
                "`google-genai` сан суугаагүй байна. `pip install -r requirements.txt` ажиллуулна уу.") from None
        return genai.Client(api_key=self._api_key(),
                            http_options={"timeout": int(timeout_seconds * 1000)})  # SDK expects milliseconds

    # -- SDK call wrapper: client lifecycle + error mapping ------------------------ #
    def _call(self, fn: Callable[[object], object], timeout_seconds: float):
        client = None
        try:
            client = self._new_client(timeout_seconds)
            return fn(client)
        except LLMError:
            raise
        except Exception as exc:  # SDK, httpx and validation errors all end up as LLMError
            raise self._map_error(exc) from None  # `from None`: the original may carry request details
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _map_error(self, exc: Exception) -> LLMError:
        code = getattr(exc, "code", None)
        if isinstance(code, int):  # google.genai.errors.APIError (duck-typed)
            detail = self._scrub(getattr(exc, "message", None) or exc)
            status = getattr(exc, "status", None) or ""
            if code in (401, 403) or (code == 400 and re.search(r"api[_ ]key", f"{status} {detail}", re.I)):
                return LLMAuthError(
                    f"Gemini API түлхүүр буруу эсвэл эрхгүй байна (HTTP {code}). {GEMINI_KEY_ENV}-г шалгана уу; "
                    "төсөл free tier-д тохирохгүй байж болзошгүй.")
            if code == 404:
                return LLMModelNotFoundError(
                    f"'{self.model}' загвар олдсонгүй эсвэл дэмжигдэхгүй байна. GEMINI_MODEL-г "
                    "https://ai.google.dev/gemini-api/docs/models хаягаар шалгана уу.")
            if code == 429:
                return LLMRateLimitError(
                    "Gemini-ийн хүсэлтийн хязгаар/квот хэтэрлээ (HTTP 429). Түр хүлээгээд дахин оролдоно уу "
                    "(хязгаарыг Google AI Studio-оос харна уу).")
            if code in (408, 504):
                return LLMTimeoutError(f"Gemini хугацаандаа хариулсангүй (HTTP {code}).")
            if code >= 500:
                return LLMResponseError(
                    f"Gemini үйлчилгээ түр ажиллахгүй байна (HTTP {code} {status}). Дахин оролдоно уу.")
            return LLMResponseError(f"Gemini хүсэлтийг татгалзлаа (HTTP {code} {status}): {detail}")

        try:
            import httpx
            timeout_types: tuple = (httpx.TimeoutException, TimeoutError)
            transport_types: tuple = (httpx.TransportError, ConnectionError, OSError)
        except ImportError:  # httpx ships with google-genai; keep the mapping usable without it
            timeout_types, transport_types = (TimeoutError,), (ConnectionError, OSError)
        if isinstance(exc, timeout_types):
            return LLMTimeoutError(
                f"Gemini {self.config.timeout_seconds:.0f} секундэд хариулсангүй. "
                "FRAME_TIMEOUT_SECONDS-ийг нэмж үзнэ үү.")
        if isinstance(exc, transport_types):
            return LLMConnectionError("Gemini API руу холбогдож чадсангүй. Интернет холболтоо шалгана уу.")
        return LLMResponseError(f"Gemini дуудлага амжилтгүй ({type(exc).__name__}): {self._scrub(exc)}")

    # -- LLMProvider ---------------------------------------------------------------- #
    def check_status(self) -> ProviderStatus:
        """Missing configuration and a confirmed connection are reported differently.

        Uses `models.get` (key + model lookup): it generates no output. It does not
        prove that quota or free-tier access is available for generation.
        """
        problem = self._config_error()
        if problem is not None:
            return ProviderStatus(False, f"Тохиргоо дутуу: {problem}")
        ok_message = f"Gemini · {self.model} — API түлхүүр ба загвар баталгаажсан"
        if time.monotonic() < self._status_ok_until:
            return ProviderStatus(True, ok_message)
        try:
            self._call(lambda client: client.models.get(model=self.model), _GEMINI_STATUS_TIMEOUT)
        except LLMError as exc:
            return ProviderStatus(False, str(exc))
        self._status_ok_until = time.monotonic() + self._STATUS_OK_TTL
        return ProviderStatus(True, ok_message)

    def chat(self, messages, *, json_mode=False, temperature=None) -> str:
        problem = self._config_error()
        if problem is not None:
            raise problem
        system, contents = to_gemini_contents(messages)

        config: dict = {}
        if system:
            config["system_instruction"] = system
        if json_mode:
            config["response_mime_type"] = "application/json"
        if self.config.gemini_temperature is not None:  # see class docstring
            config["temperature"] = self.config.gemini_temperature

        response = self._call(
            lambda client: client.models.generate_content(
                model=self.model, contents=contents, config=config or None),
            self.config.timeout_seconds)
        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        feedback = getattr(response, "prompt_feedback", None)
        block = getattr(feedback, "block_reason", None)
        if block:
            raise LLMResponseError(
                f"Gemini хүсэлтийг блоклосон ({getattr(block, 'name', block)}). Агуулгаа өөрчилж үзнэ үү.")
        try:
            text = response.text
        except Exception:
            text = None
        if isinstance(text, str) and text.strip():
            return text
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        raise LLMResponseError(
            f"Gemini хоосон хариу буцаалаа (finish_reason={getattr(reason, 'name', reason)}). "
            "Дахин оролдох эсвэл агуулгыг өөрчилнө үү.")


PROVIDERS: dict[str, Callable[[LLMConfig], LLMProvider]] = {
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
}


def create_provider(config: LLMConfig | None = None) -> LLMProvider:
    config = config or LLMConfig.from_env()
    try:
        factory = PROVIDERS[config.provider]
    except KeyError:
        raise ValueError(
            f"Unknown FRAME_LLM_PROVIDER '{config.provider}'. Available: {', '.join(PROVIDERS)}"
        ) from None
    return factory(config)


# --------------------------------------------------------------------------- #
# Structured output: extract JSON, validate with Pydantic, one controlled repair
# --------------------------------------------------------------------------- #
def extract_json(text: str):
    """Parse a JSON value from model text, tolerating code fences and chatter."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start:end + 1])  # may raise JSONDecodeError (a ValueError)
    raise ValueError("No JSON object found in the model output.")


def _describe_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in exc.errors()[:8]]
        return "; ".join(parts)
    return str(exc)


T = TypeVar("T", bound=BaseModel)


def generate_validated(
    llm: LLMProvider,
    messages: list[dict],
    model_cls: type[T],
    *,
    validator: Callable[[T], None] | None = None,
    max_repairs: int = 1,
    temperature: float | None = None,
) -> T:
    """Ask the model for JSON and return a validated `model_cls` instance.

    On invalid JSON / schema errors / `validator` failures the model gets exactly
    `max_repairs` chances to fix its answer. Afterwards `InvalidModelOutputError`
    is raised: structured data is never fabricated.
    """
    raw = llm.chat(messages, json_mode=True, temperature=temperature)
    attempt = 0
    while True:
        try:
            result = model_cls.model_validate(extract_json(raw))
            if validator is not None:
                validator(result)
            return result
        except ValueError as exc:  # JSONDecodeError and pydantic ValidationError are ValueErrors
            problem = _describe_error(exc)
            if attempt >= max_repairs:
                raise InvalidModelOutputError(
                    f"Загварын гаргасан JSON буруу хэвээр байна ({problem}). "
                    "Дахин оролдох эсвэл илүү том загвар ашиглана уу.",
                    raw=raw[:2000],
                ) from exc
            attempt += 1
            repair = messages + [
                {"role": "assistant", "content": raw[:6000]},
                {"role": "user", "content": prompts.repair_prompt(problem)},
            ]
            raw = llm.chat(repair, json_mode=True, temperature=0.1)
