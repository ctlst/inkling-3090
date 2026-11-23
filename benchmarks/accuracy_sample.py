"""Prepare or score a pinned, deterministic MMLU-Pro sample. See benchmarks/accuracy.md for the protocol."""

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

TEST_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
SUBJECTS = ("business", "economics", "history", "math", "philosophy", "psychology")


def prepare(parquet, output):
    import pyarrow.parquet as pq

    if hashlib.sha256(parquet.read_bytes()).hexdigest() != TEST_SHA256:
        raise ValueError(
            "Expected the pinned MMLU-Pro test parquet listed in benchmarks/accuracy.md"
        )
    rows = pq.read_table(parquet).to_pylist()
    sample = []
    for subject in SUBJECTS:
        group = [r for r in rows if r["category"] == subject]
        group.sort(
            key=lambda r: hashlib.sha256(
                f"inkling-419/{r['question_id']}".encode()
            ).digest()
        )
        assert len(group) >= 4
        for row in group[:4]:
            options = "\n".join(
                f"{chr(65 + i)}. {v}" for i, v in enumerate(row["options"])
            )
            prompt = f"{row['question']}\n\n{options}\n\nReturn only the letter of the correct option."
            sample.append(
                {
                    "id": row["question_id"],
                    "subject": subject,
                    "prompt": prompt,
                    "gold": row["answer"],
                }
            )
    output.write_text(json.dumps(sample))


def score_item(url, item):
    body = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "seed": 419,
        "max_tokens": 2048,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=900) as resp:
        result = json.load(resp)
    choice = result["choices"][0]
    answer = (choice["message"].get("content") or "").strip()
    match = re.fullmatch(r"\(?([A-J])\)?[.\s]*", answer)
    letter = match[1] if match else None
    return {
        "id": item["id"],
        "subject": item["subject"],
        "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
        "gold": item["gold"],
        "answer": answer,
        "letter": letter,
        "correct": letter == item["gold"],
        "finish": choice["finish_reason"],
        "usage": result["usage"],
    }


def evaluate(url, fixtures, output, concurrency=1):
    items = json.loads(fixtures.read_text())
    completed = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        jobs = {pool.submit(score_item, url, item): i for i, item in enumerate(items)}
        for job in as_completed(jobs):
            row = job.result()
            completed[jobs[job]] = row
            ordered = [completed[i] for i in sorted(completed)]
            output.write_text(json.dumps(ordered, indent=2) + "\n")
            print(
                f"{len(ordered)} / {len(items)}: {row['subject']} {row['id']} {row['letter']} gold {row['gold']}",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("parquet", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("url")
    run.add_argument("fixtures", type=Path)
    for cmd in (prep, run):
        cmd.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.parquet, args.output)
    else:
        evaluate(args.url, args.fixtures, args.output, args.concurrency)
