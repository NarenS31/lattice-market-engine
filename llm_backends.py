"""
Model-agnostic decision backends for the LLM trading agents.

The agent layer no longer talks to any vendor SDK directly.  It builds a
(system_prompt, user_prompt) pair and hands it to an `LLMBackend`, which
returns the parsed decision dict:

    {"action": "buy"|"sell"|"hold",
     "quantity": <number>,
     "limit_price": <number or null>,
     "reasoning": "<one sentence>"}

Two backends ship here:

  AnthropicBackend  — the original Claude path (prompt-cached system prompt).
  OllamaBackend     — a local Ollama server, with schema-checked JSON parsing
                      and bounded retries using a stricter re-ask.

Every parse failure is appended to a JSONL log (LATTICE_PARSE_LOG, default
``parse_failures.jsonl``) so the failure rate of a given model is measurable
rather than invisible.

Selection is config-driven — see `backend_for()` and `configure_backends()`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Defaults (all overridable by environment variable)
# ---------------------------------------------------------------------------

DEFAULT_ANTHROPIC_MODEL = os.environ.get(
    "LATTICE_LLM_MODEL", "claude-sonnet-4-20250514"
)
DEFAULT_OLLAMA_MODEL = os.environ.get("LATTICE_OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_OLLAMA_HOST = os.environ.get("LATTICE_OLLAMA_HOST", "http://localhost:11434")
DEFAULT_PARSE_LOG = os.environ.get("LATTICE_PARSE_LOG", "parse_failures.jsonl")

MAX_TOKENS = 256

# Archetype keys used for per-archetype backend selection.
ARCHETYPES = ("retail", "institutional", "quant")


# ---------------------------------------------------------------------------
# Decision schema
# ---------------------------------------------------------------------------

_VALID_ACTIONS = ("buy", "sell", "hold")


def strip_code_fence(text: str) -> str:
    """Remove a leading ```/```json fence and its closing fence, if present."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
    return "\n".join(lines[1:end]).strip()


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} span in `text`, or None.

    Small models frequently wrap the object in prose ("Sure! Here is the
    JSON: {...} Let me know if..."), which `json.loads` rejects outright.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def validate_decision(obj: object) -> tuple[bool, str]:
    """Check `obj` against the decision schema. Returns (ok, reason)."""
    if not isinstance(obj, dict):
        return False, f"not_an_object:{type(obj).__name__}"

    action = obj.get("action")
    if not isinstance(action, str):
        return False, "action_missing_or_not_string"
    if action.lower().strip() not in _VALID_ACTIONS:
        return False, f"action_invalid:{action[:32]}"

    # quantity may be absent/null on a hold, but if present must be numeric.
    qty = obj.get("quantity")
    if qty is not None and not isinstance(qty, (int, float)):
        try:
            float(qty)
        except (TypeError, ValueError):
            return False, "quantity_not_numeric"

    lp = obj.get("limit_price")
    if lp is not None and not isinstance(lp, (int, float)):
        try:
            float(lp)
        except (TypeError, ValueError):
            return False, "limit_price_not_numeric"

    if "reasoning" in obj and not isinstance(obj["reasoning"], str):
        return False, "reasoning_not_string"

    return True, ""


def parse_decision(text: str) -> tuple[Optional[dict], str]:
    """Parse raw model output into a decision dict.

    Returns (decision, reason). `decision` is None on failure and `reason`
    is a short machine-readable failure tag ("" on success).
    """
    if not text or not text.strip():
        return None, "empty_response"

    candidate = strip_code_fence(text)
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        span = _extract_json_object(candidate)
        if span is None:
            return None, "no_json_object_found"
        try:
            obj = json.loads(span)
        except (json.JSONDecodeError, ValueError):
            return None, "malformed_json"

    ok, reason = validate_decision(obj)
    if not ok:
        return None, reason
    return obj, ""


# ---------------------------------------------------------------------------
# Failure logging
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()
_stats_lock = threading.Lock()
_STATS: dict[str, dict[str, int]] = {}

_RAW_SNIPPET_CHARS = 600


def _bump(backend: str, key: str) -> None:
    with _stats_lock:
        _STATS.setdefault(backend, {"attempts": 0, "successes": 0, "failures": 0})
        _STATS[backend][key] += 1


def stats() -> dict[str, dict[str, int]]:
    """Return a snapshot of per-backend attempt/success/failure counters."""
    with _stats_lock:
        return {k: dict(v) for k, v in _STATS.items()}


def reset_stats() -> None:
    with _stats_lock:
        _STATS.clear()


def log_parse_failure(
    backend: str,
    model: str,
    attempt: int,
    reason: str,
    raw: str,
    path: Optional[str] = None,
) -> None:
    """Append one parse failure to the JSONL failure log (best effort)."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "model": model,
        "attempt": attempt,
        "reason": reason,
        "raw": (raw or "")[:_RAW_SNIPPET_CHARS],
        "raw_len": len(raw or ""),
    }
    target = path or DEFAULT_PARSE_LOG
    try:
        with _log_lock:
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
    except OSError:
        # Never let telemetry break a simulation run.
        pass


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

_STRICT_SUFFIX = (
    "\n\nYOUR PREVIOUS RESPONSE WAS REJECTED: {reason}.\n"
    "Return ONLY valid JSON. No prose, no explanation, no markdown fences. "
    "Your entire response must be exactly one JSON object of this shape:\n"
    '{{"action": "buy"|"sell"|"hold", "quantity": <number>, '
    '"limit_price": <number or null>, "reasoning": "<one short sentence>"}}'
)


class LLMBackend(ABC):
    """A source of trading decisions.

    Implementations own transport, retries, and parsing. `get_decision`
    returns a validated decision dict, or None if no usable decision could
    be obtained — callers treat None as "hold this tick".
    """

    name: str = "base"

    @property
    def model(self) -> str:
        """The model identifier this backend will call."""
        raise NotImplementedError

    @abstractmethod
    def get_decision(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        ...

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class AnthropicBackend(LLMBackend):
    """The original Claude path, using the official `anthropic` SDK.

    The archetype system prompt is sent as a cached block, exactly as before,
    so repeated per-tick calls hit the prompt cache.

    `model_resolver` exists so `agents._MODEL` stays authoritative and
    monkeypatchable — `experiments/model_scale.py` swaps that global between
    runs to compare model tiers.
    """

    name = "anthropic"

    def __init__(
        self,
        model: Optional[str] = None,
        model_resolver: Optional[Callable[[], str]] = None,
        max_attempts: int = 1,
        max_tokens: int = MAX_TOKENS,
        log_path: Optional[str] = None,
    ) -> None:
        self._model = model
        self._model_resolver = model_resolver
        self.max_attempts = max(1, max_attempts)
        self.max_tokens = max_tokens
        self.log_path = log_path
        self._client = None
        self._client_error: Optional[str] = None

    @property
    def model(self) -> str:
        if self._model is not None:
            return self._model
        if self._model_resolver is not None:
            return self._model_resolver()
        return DEFAULT_ANTHROPIC_MODEL

    def _get_client(self):
        """Construct the SDK client lazily so import never requires a key."""
        if self._client is not None:
            return self._client
        if self._client_error is not None:
            return None
        try:
            import anthropic

            self._client = anthropic.Anthropic()
        except Exception as exc:  # missing package or unresolvable credentials
            self._client_error = f"{type(exc).__name__}: {exc}"
            return None
        return self._client

    @property
    def client_error(self) -> Optional[str]:
        """Why the client could not be built, if it could not be."""
        self._get_client()
        return self._client_error

    def _one_call(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        if client is None:
            raise RuntimeError(self._client_error or "anthropic client unavailable")
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    def get_decision(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        return _run_with_retries(
            backend=self,
            call=self._one_call,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_attempts=self.max_attempts,
            log_path=self.log_path,
        )


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    """A local Ollama server.

    Uses the `ollama` Python package when importable and falls back to raw
    HTTP against ``POST {host}/api/generate`` otherwise.

    Small local models are far less reliable at emitting bare JSON than a
    frontier model, so this backend retries up to `max_attempts` times,
    appending an increasingly blunt "return ONLY valid JSON" instruction and
    logging every rejected response.

    Set `json_mode=True` to pass Ollama's native ``format="json"`` constraint.
    That raises the parse rate sharply, but leave it off for the pilot — the
    point of the pilot is to measure how the model behaves unconstrained.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        max_attempts: int = 3,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.7,
        json_mode: bool = False,
        timeout: float = 120.0,
        log_path: Optional[str] = None,
    ) -> None:
        self._model = model
        self.host = host.rstrip("/")
        self.max_attempts = max(1, max_attempts)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self.timeout = timeout
        self.log_path = log_path
        self._client = None
        self._use_pkg: Optional[bool] = None

    @property
    def model(self) -> str:
        return self._model

    def _ensure_transport(self) -> None:
        if self._use_pkg is not None:
            return
        try:
            import ollama

            self._client = ollama.Client(host=self.host, timeout=self.timeout)
            self._use_pkg = True
        except Exception:
            self._client = None
            self._use_pkg = False

    def _options(self) -> dict:
        return {"num_predict": self.max_tokens, "temperature": self.temperature}

    def _one_call(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_transport()

        if self._use_pkg:
            kwargs = dict(
                model=self.model,
                system=system_prompt,
                prompt=user_prompt,
                stream=False,
                options=self._options(),
            )
            if self.json_mode:
                kwargs["format"] = "json"
            resp = self._client.generate(**kwargs)
            # ollama>=0.4 returns a pydantic model; older versions a dict.
            return getattr(resp, "response", None) or resp["response"]

        import requests

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": self._options(),
        }
        if self.json_mode:
            payload["format"] = "json"
        r = requests.post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json().get("response", "")

    def get_decision(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        return _run_with_retries(
            backend=self,
            call=self._one_call,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_attempts=self.max_attempts,
            log_path=self.log_path,
        )


# ---------------------------------------------------------------------------
# Shared call / retry / logging loop
# ---------------------------------------------------------------------------

def _run_with_retries(
    backend: LLMBackend,
    call: Callable[[str, str], str],
    system_prompt: str,
    user_prompt: str,
    max_attempts: int,
    log_path: Optional[str],
) -> Optional[dict]:
    """Call `call` until it yields a schema-valid decision or attempts run out.

    Each retry appends a stricter JSON-only instruction to the user prompt.
    Transport errors and parse failures are both logged and both consume an
    attempt.
    """
    model = backend.model
    prompt = user_prompt

    for attempt in range(1, max_attempts + 1):
        _bump(backend.name, "attempts")
        try:
            raw = call(system_prompt, prompt)
        except Exception as exc:
            reason = f"transport_error:{type(exc).__name__}"
            _bump(backend.name, "failures")
            log_parse_failure(
                backend.name, model, attempt, reason, str(exc)[:_RAW_SNIPPET_CHARS],
                log_path,
            )
            # A transport failure will not be fixed by a sterner prompt.
            return None

        decision, reason = parse_decision(raw)
        if decision is not None:
            _bump(backend.name, "successes")
            return decision

        _bump(backend.name, "failures")
        log_parse_failure(backend.name, model, attempt, reason, raw, log_path)
        prompt = user_prompt + _STRICT_SUFFIX.format(reason=reason)

    return None


# ---------------------------------------------------------------------------
# Configuration / selection
# ---------------------------------------------------------------------------

_default_kind = os.environ.get("LATTICE_BACKEND", "anthropic").strip().lower()
_overrides: dict[str, str] = {}
_cache: dict[tuple, LLMBackend] = {}
_anthropic_model_resolver: Optional[Callable[[], str]] = None


def set_anthropic_model_resolver(fn: Optional[Callable[[], str]]) -> None:
    """Install a callable that yields the current Anthropic model name.

    `agents` uses this to keep its module-level `_MODEL` authoritative, so
    experiments that monkeypatch `agents._MODEL` still work.
    """
    global _anthropic_model_resolver
    _anthropic_model_resolver = fn
    _cache.clear()


def build_backend(kind: str, **kwargs) -> LLMBackend:
    """Construct a backend by name ("anthropic" | "ollama")."""
    k = kind.strip().lower()
    if k == "anthropic":
        kwargs.setdefault("model_resolver", _anthropic_model_resolver)
        return AnthropicBackend(**kwargs)
    if k == "ollama":
        return OllamaBackend(**kwargs)
    raise ValueError(
        f"Unknown backend {kind!r}. Expected one of: anthropic, ollama."
    )


def configure_backends(
    default: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> None:
    """Set the process-wide default backend and per-archetype overrides.

    `overrides` maps an archetype key ("retail"/"institutional"/"quant") to a
    backend name. Called by `apply_cli_args`, or directly from a script.
    """
    global _default_kind
    if default:
        _default_kind = default.strip().lower()
    if overrides:
        for k, v in overrides.items():
            if v:
                _overrides[k.strip().lower()] = v.strip().lower()
    _cache.clear()


def resolve_kind(archetype: Optional[str]) -> str:
    """Which backend name should `archetype` use?

    Precedence: explicit override > LATTICE_BACKEND_<ARCHETYPE> env >
    process default (LATTICE_BACKEND env, or "anthropic").
    """
    if archetype:
        key = archetype.strip().lower()
        if key in _overrides:
            return _overrides[key]
        env = os.environ.get(f"LATTICE_BACKEND_{key.upper()}")
        if env:
            return env.strip().lower()
    return _default_kind


def backend_for(archetype: Optional[str] = None) -> LLMBackend:
    """Return the (cached) backend instance for an archetype."""
    kind = resolve_kind(archetype)
    key = (kind, DEFAULT_OLLAMA_MODEL if kind == "ollama" else None)
    if key not in _cache:
        _cache[key] = build_backend(kind)
    return _cache[key]


def add_backend_cli_args(parser) -> None:
    """Register --backend and per-archetype --backend-<archetype> flags."""
    parser.add_argument(
        "--backend",
        choices=["anthropic", "ollama"],
        default=None,
        help="Decision backend for all LLM agents (default: $LATTICE_BACKEND "
             "or anthropic).",
    )
    for arch in ARCHETYPES:
        parser.add_argument(
            f"--backend-{arch}",
            choices=["anthropic", "ollama"],
            default=None,
            help=f"Override the backend for the {arch} archetype only.",
        )


def apply_cli_args(args) -> None:
    """Apply flags registered by `add_backend_cli_args`."""
    overrides = {
        arch: getattr(args, f"backend_{arch}", None) for arch in ARCHETYPES
    }
    configure_backends(default=getattr(args, "backend", None), overrides=overrides)
