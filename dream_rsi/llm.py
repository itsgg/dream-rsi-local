"""Chat and embedding clients, no SDK.

Two providers. Ollama's native API is the default and keeps `num_ctx`, a real sampling seed and qwen3's
`think: false`. Setting a base URL switches to any OpenAI-compatible endpoint (OpenAI, Anthropic's compat
layer, Together, vLLM, LiteLLM, ...), which is how you point the policy writer at a frontier model.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
BASE_URL = os.environ.get("DREAM_RSI_BASE_URL") or None
API_KEY = os.environ.get("DREAM_RSI_API_KEY") or None
EMBED_BASE_URL = os.environ.get("DREAM_RSI_EMBED_BASE_URL") or None
EMBED_API_KEY = os.environ.get("DREAM_RSI_EMBED_API_KEY") or None

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def normalise_base(url: str | None) -> str | None:
    """OpenAI-compatible bases carry a version path. Tolerate the common `https://host` with none."""
    if not url:
        return None
    url = url.rstrip("/")
    return url + "/v1" if urlparse(url).path in ("", "/") else url


class _Http:
    """POST JSON with retries, and learn which optional parameters this endpoint rejects.

    Providers disagree about `temperature`, `seed` and `max_tokens` (reasoning models reject the first two,
    newer OpenAI models want `max_completion_tokens`). Rather than a config knob per provider, a 400 that
    names one of them drops or renames it and retries; everything else fails fast instead of burning five
    backoffs on a bad key.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.drop: set[str] = set()
        self.max_field = "max_tokens"

    def _adapt(self, detail: str) -> bool:
        d = detail.lower()
        if "max_completion_tokens" in d and self.max_field == "max_tokens":
            self.max_field = "max_completion_tokens"
            return True
        for p in ("temperature", "seed", "top_p"):
            if p in d and p not in self.drop:
                self.drop.add(p)
                return True
        return False

    def post(self, url: str, body: dict, timeout: float, retries: int = 5) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(retries):
            payload = {k: v for k, v in body.items() if k not in self.drop}
            if "max_tokens" in body and self.max_field != "max_tokens":
                payload.pop("max_tokens", None)
                payload[self.max_field] = body["max_tokens"]
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:   # must precede URLError, it is a subclass
                detail = e.read().decode("utf-8", "replace")[:400]
                if e.code == 400 and self._adapt(detail):
                    continue                      # retry immediately with the offending parameter removed
                if e.code not in RETRY_STATUS:
                    raise RuntimeError(f"{url} returned HTTP {e.code}: {detail}") from None
                last = f"HTTP {e.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.HTTPException,
                    ConnectionError, OSError) as e:
                last = e                          # Ollama drops connections under memory pressure
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"{url} failed after {retries} attempts: {last}")


class LLM:
    def __init__(self, model: str, num_ctx: int = 8192, temperature: float = 0.8,
                 num_predict: int = 1200, log_path: str | None = None, seed: int | None = None,
                 base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.num_predict = num_predict
        self.seed = seed
        self.base_url = normalise_base(base_url)
        self._http = _Http(api_key)
        self._lock = threading.Lock()
        self._seq = 0  # per-call sequence number, assigned at request time so parallel calls get distinct seeds
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0
        self.log_path = log_path

    @property
    def provider(self) -> str:
        return "openai-compatible" if self.base_url else "ollama"

    def describe(self) -> str:
        return f"{self.model} via {self.provider} at {self.base_url or OLLAMA_URL}"

    def chat(self, system: str, user: str, temperature: float | None = None,
             num_predict: int | None = None, tag: str = "", seed_key: int | None = None) -> str:
        """`seed_key` makes the sampling seed a function of the caller's position rather than of arrival
        order, which is what parallel online attempts need to be reproducible under --seed. Note that an
        OpenAI-compatible endpoint treats `seed` as best-effort at most, so runs there are not reproducible."""
        with self._lock:
            seq = self._seq
            self._seq += 1
        key = seq if seed_key is None else seed_key
        temp = self.temperature if temperature is None else temperature
        npred = self.num_predict if num_predict is None else num_predict
        seed = None if self.seed is None else (self.seed * 1_000_003 + key) % (2 ** 31 - 1)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        if self.base_url:
            url = f"{self.base_url}/chat/completions"
            body = {"model": self.model, "messages": messages, "temperature": temp, "max_tokens": npred,
                    **({"seed": seed} if seed is not None else {})}
        else:
            url = f"{OLLAMA_URL}/api/chat"
            body = {"model": self.model, "stream": False,
                    # qwen3 models default to a long "thinking" phase; keep replies fast and short
                    **({"think": False} if self.model.startswith("qwen3") else {}),
                    "messages": messages,
                    "options": {"temperature": temp, "num_ctx": self.num_ctx, "num_predict": npred,
                                **({"seed": seed} if seed is not None else {})}}

        t0 = time.time()
        data = self._http.post(url, body, timeout=600)
        dt = time.time() - t0

        if self.base_url:
            choices = data.get("choices")
            if not choices:
                raise RuntimeError(f"{url} returned no choices: {str(data)[:300]}")
            text = choices[0].get("message", {}).get("content") or ""
            usage = data.get("usage") or {}
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        else:
            if "message" not in data:
                raise RuntimeError(f"ollama returned an error payload: {str(data)[:300]}")
            text = data["message"]["content"]
            pt, ct = data.get("prompt_eval_count"), data.get("eval_count")

        with self._lock:
            self.calls += 1
            self.seconds += dt
            self.prompt_tokens += int(pt or 0)
            self.completion_tokens += int(ct or 0)
            if self.log_path:
                with open(self.log_path, "a") as f:   # the api key is never part of this record
                    f.write(json.dumps({"tag": tag, "model": self.model, "provider": self.provider,
                                        "seconds": round(dt, 1), "seq": seq, "seed_key": key, "seed": seed,
                                        "prompt_tokens": pt, "completion_tokens": ct,
                                        "system": system, "user": user, "response": text}) + "\n")
        return text

    def stats(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self.seconds, 1)}


def extract_code(text: str) -> str | None:
    """Return the longest fenced code block (python or unlabelled), or None."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        return None
    blocks.sort(key=len)
    return blocks[-1].strip() + "\n"


class Embedder:
    """Embeddings with an in-memory cache, used for novelty rejection.

    Defaults to Ollama even when chat runs against a remote endpoint: embeddings are cheap, called often,
    and there is no reason to pay an API for them.
    """

    def __init__(self, model: str = "bge-m3", base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = normalise_base(base_url)
        self._http = _Http(api_key)
        self._cache: dict[str, list[float]] = {}
        self.calls = 0

    @property
    def provider(self) -> str:
        return "openai-compatible" if self.base_url else "ollama"

    def describe(self) -> str:
        return f"{self.model} via {self.provider} at {self.base_url or OLLAMA_URL}"

    @staticmethod
    def normalize(code: str) -> str:
        lines = []
        for l in code.splitlines():
            l = re.sub(r"#.*", "", l).rstrip()
            if l.strip():
                lines.append(re.sub(r"\s+", " ", l.strip()))
        return "\n".join(lines)

    def embed(self, code: str) -> list[float]:
        key = self.normalize(code)
        if key in self._cache:
            return self._cache[key]
        if self.base_url:
            data = self._http.post(f"{self.base_url}/embeddings",
                                   {"model": self.model, "input": key}, timeout=120)
            vec = data["data"][0]["embedding"]
        else:
            data = self._http.post(f"{OLLAMA_URL}/api/embed",
                                   {"model": self.model, "input": key}, timeout=120)
            vec = data["embeddings"][0]
        self.calls += 1
        self._cache[key] = vec
        return vec

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def max_similarity(self, code: str, others: list[str]) -> tuple[float, int]:
        """Highest cosine similarity of `code` to any of `others`, and its index (-1 if none)."""
        if not others:
            return 0.0, -1
        v = self.embed(code)
        best, idx = -1.0, -1
        for i, o in enumerate(others):
            s = self.cosine(v, self.embed(o))
            if s > best:
                best, idx = s, i
        return best, idx
