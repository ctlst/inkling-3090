"""Run the release's public LongBench v2 retrieval fixture against a local endpoint."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

SAMPLE = "66ec3fa7821e116aacb1c75d"
PROMPT_SHA256 = "3e970c803f439496aa0d5164de3bcc01ea9513960780efca68aacadc85624cd3"


def prepare(dataset, template):
    row = next(r for r in json.loads(dataset.read_bytes()) if r["_id"] == SAMPLE)
    prompt = template.read_text().replace("$DOC$", row["context"].strip())
    prompt = prompt.replace("$Q$", row["question"].strip())
    for letter in "ABCD":
        prompt = prompt.replace(f"$C_{letter}$", row[f"choice_{letter}"].strip())
    if hashlib.sha256(prompt.encode()).hexdigest() != PROMPT_SHA256:
        raise ValueError(
            "LongBench question or template differs from the release fixture"
        )
    return prompt, row["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("url")
    args = parser.parse_args()
    prompt, gold = prepare(args.dataset, args.template)
    body = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "seed": 419,
        "max_tokens": 1536,
        "chat_template_kwargs": {"reasoning_effort": "none"},
        "cache_salt": "official-guide-419",
    }
    req = Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=900) as resp:
        response = json.load(resp)
    choice = response["choices"][0]
    final = (choice["message"].get("content") or "").strip()
    match = re.search(
        r"The correct answer is\s*\(?([A-D])\b", final.replace("*", ""), re.IGNORECASE
    )
    answer = match[1].upper() if match else None
    passed = (
        answer == gold
        and choice["finish_reason"] == "stop"
        and response["usage"]["prompt_tokens"] == 195015
    )
    print(
        json.dumps(
            {
                "id": SAMPLE,
                "prompt_sha256": PROMPT_SHA256,
                "usage": response["usage"],
                "finish": choice["finish_reason"],
                "gold": gold,
                "final": final,
                "passed": passed,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
