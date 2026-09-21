"""OpenAI-compatible client paths, against a mock server. No real API, no LLM."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dream_rsi.llm import LLM, Embedder, normalise_base

assert normalise_base("https://api.openai.com") == "https://api.openai.com/v1"
assert normalise_base("https://api.openai.com/v1/") == "https://api.openai.com/v1"
assert normalise_base("http://localhost:8000/openai/v1") == "http://localhost:8000/openai/v1"
assert normalise_base(None) is None
print("base url normalisation OK")

seen: list[dict] = []
script: list[tuple[int, str]] = []          # queued (status, detail) failures, popped per request


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        if script:
            code, detail = script.pop(0)
            payload = json.dumps({"error": {"message": detail}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.endswith("/embeddings"):
            out = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        else:
            out = {"choices": [{"message": {"content": "hello from the mock"}}],
                   "usage": {"prompt_tokens": 11, "completion_tokens": 3}}
        payload = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}/v1"

# --- a plain call ---------------------------------------------------------------------------------------
llm = LLM("gpt-frontier", base_url=BASE, api_key="sk-test", seed=7)
assert llm.provider == "openai-compatible", llm.provider
out = llm.chat("sys", "user", tag="t", seed_key=3)
assert out == "hello from the mock", out
req = seen[-1]
assert req["path"] == "/v1/chat/completions", req["path"]
assert req["auth"] == "Bearer sk-test"
assert req["body"]["messages"][0] == {"role": "system", "content": "sys"}
assert "max_tokens" in req["body"] and "seed" in req["body"]
assert llm.stats() == {"calls": 1, "prompt_tokens": 11, "completion_tokens": 3,
                       "seconds": llm.stats()["seconds"]}
print("openai chat OK (auth header, message shape, usage accounting)")

# --- the endpoint rejects parameters, one at a time -----------------------------------------------------
seen.clear()
script[:] = [(400, "Unsupported parameter: 'max_tokens' is not supported, use 'max_completion_tokens'"),
             (400, "Unsupported value: 'temperature' does not support 0.8 with this model")]
llm2 = LLM("reasoning-model", base_url=BASE, api_key="k", seed=1)
assert llm2.chat("s", "u") == "hello from the mock"
assert len(seen) == 3, f"expected two rejections then a success, got {len(seen)} requests"
final = seen[-1]["body"]
assert "max_tokens" not in final and final.get("max_completion_tokens") == 1200, final
assert "temperature" not in final, final
# the adaptation is remembered, so the next call is shaped correctly first time
llm2.chat("s", "u")
assert len(seen) == 4 and "temperature" not in seen[-1]["body"], seen[-1]["body"]
print("parameter adaptation OK (max_completion_tokens, temperature dropped, remembered)")

# --- a bad key fails immediately rather than burning five backoffs ---------------------------------------
seen.clear()
script[:] = [(401, "Incorrect API key provided")]
try:
    LLM("m", base_url=BASE, api_key="bad").chat("s", "u")
    raise AssertionError("a 401 should raise")
except RuntimeError as e:
    assert "401" in str(e) and "Incorrect API key" in str(e), e
assert len(seen) == 1, f"a 401 must not be retried, saw {len(seen)} requests"
print("auth failure OK (fails fast, message preserved)")

# --- embeddings -----------------------------------------------------------------------------------------
seen.clear()
emb = Embedder("text-embed", base_url=BASE, api_key="k")
v = emb.embed("def priority(now, last_used, freq, inserted):\n    return last_used\n")
assert v == [0.1, 0.2, 0.3], v
assert seen[-1]["path"] == "/v1/embeddings", seen[-1]["path"]
emb.embed("def priority(now, last_used, freq, inserted):\n    return last_used\n")
assert len(seen) == 1, "identical source should hit the cache"
print("openai embeddings OK (path, cache)")

# --- the api key never reaches the call log -------------------------------------------------------------
import tempfile, pathlib
logf = pathlib.Path(tempfile.mkdtemp()) / "calls.jsonl"
script.clear()
LLM("m", base_url=BASE, api_key="sk-super-secret", log_path=str(logf)).chat("s", "u", tag="x")
text = logf.read_text()
assert "sk-super-secret" not in text, "the api key leaked into llm_calls.jsonl"
assert json.loads(text)["provider"] == "openai-compatible"
print("call log OK (no key, provider recorded)")

srv.shutdown()
print("OK")
