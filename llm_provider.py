"""LLM provider adapter.

`LLMProvider` is the only thing the agent and tools know about. Ollama is the
default implementation; another HTTP API can be supported by subclassing
`LLMProvider` and registering it in `PROVIDERS` (see README, "Adding a provider").

Nothing in this module fabricates model output: if the model cannot be reached,
a typed `LLMError` is raised and surfaced to the user.
"""
from __future__ import annotations

import json
import os
import re
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


class InvalidModelOutputError(LLMError):
    """The model's output stayed invalid after the allowed repair attempt."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_env_file(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    timeout_seconds: float = 300.0  # CPU-only inference can be slow
    temperature: float = 0.4
    num_ctx: int = 8192

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LLMConfig":
        env = os.environ if env is None else env
        default = cls()

        def number(key: str, fallback, cast):
            try:
                return cast(env[key]) if key in env else fallback
            except ValueError:
                return fallback

        return cls(
            provider=env.get("FRAME_LLM_PROVIDER", default.provider).strip().lower(),
            base_url=env.get("OLLAMA_BASE_URL", default.base_url).strip().rstrip("/"),
            model=env.get("FRAME_MODEL", default.model).strip(),
            timeout_seconds=number("FRAME_TIMEOUT_SECONDS", default.timeout_seconds, float),
            temperature=number("FRAME_TEMPERATURE", default.temperature, float),
            num_ctx=number("FRAME_NUM_CTX", default.num_ctx, int),
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


PROVIDERS: dict[str, Callable[[LLMConfig], LLMProvider]] = {
    "ollama": OllamaProvider,
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
