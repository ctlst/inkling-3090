"""Exercise the model registry and a deterministic chat turn on a running endpoint."""
import argparse
import json
from urllib.request import Request, urlopen

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
    base = parser.parse_args().url.rstrip("/")
    with urlopen(base + "/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    model = next((m for m in models if m["id"] == "inkling-2-mini"), None)
    if not model or model.get("max_model_len") != 196608:
        raise SystemExit("Expected inkling-2-mini with a 196608-token limit")
    payload = {
        "model": model["id"],
        "messages": [{"role": "user", "content": "What is 17 plus 8? Output just the integer."}],
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"reasoning_effort": "none"},
    }
    request = Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    choice = result["choices"][0]
    if choice["finish_reason"] != "stop" or (choice["message"].get("content") or "").strip() != "25":
        raise SystemExit("Chat check failed: " + json.dumps(choice))
    print("Model, context limit and arithmetic response verified.")

if __name__ == "__main__":
    main()
