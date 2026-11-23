"""Measure streamed single-user requests against an isolated Inkling endpoint."""
import argparse
import base64
import hashlib
import json
import re
import struct
import time
import uuid
import zlib
from pathlib import Path
from urllib.request import Request, urlopen


def chart():
    width, height = 320, 140
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            color = (242, 242, 242)
            for top, length, rgb in [(12, 120, (224, 45, 55)), (55, 270, (35, 90, 225)), (98, 68, (34, 161, 90))]:
                if top <= y < top + 28 and 16 <= x < 16 + length:
                    color = rgb
            row.extend(color)
        rows.append(row)

    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def user(content):
    return {"role": "user", "content": content}


def cases():
    tool = {"type": "function", "function": {"name": "find_departures", "description": "Search the local rail schedule.",
            "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in ["origin", "destination", "after"]},
                           "required": ["origin", "destination", "after"]}}}
    return {
        "chat": {"messages": [user("I'm planning a club meeting. Room A seats 12, B seats 24 and C seats 40. I want the smallest room that seats everyone."),
                              {"role": "assistant", "content": "How many people are coming?"},
                              user("There are 19 attendees plus 2 presenters. Which room should I book and how many seats will be empty? Answer conversationally in one sentence.")]},
        "prose": {"messages": [user("Turn these field notes into a clear 100–150 word update for library members. Use ordinary prose, no heading. "
                                   "The North Branch reopens on June 12 after a roof repair. Weekday hours are 9 am to 6 pm. "
                                   "The book drop remains on Cedar Street. The children's room will reopen separately on July 3. "
                                   "Make clear which spaces are available on which date. Do not invent a cause for the delay or promise new services.")]},
        "tool": {"messages": [user("Please look up the next trains from Dover to Ashford departing after 14:30. Use the schedule tool.")], "tools": [tool]},
        "vision": {"messages": [user([{"type": "text", "text": "These three horizontal bars show queue sizes. Rank the bar colors from longest to shortest. Reply with only the three colors in order."},
                                      {"type": "image_url", "image_url": {"url": chart()}}])]},
    }


def grade(name, answer, calls):
    s = answer.lower()
    if name == "chat":
        return {"room B": bool(re.search(r"\b(?:room\s+)?b\b", s)), "three seats": bool(re.search(r"\b(3|three)\b", s))}
    if name == "prose":
        return {"length": 100 <= len(answer.split()) <= 150, "branch": "north branch" in s, "roof": "roof" in s,
                "June 12": bool(re.search(r"june\s+12|12\s+june", s)), "July 3": bool(re.search(r"july\s+3|3\s+july", s)),
                "book drop": "cedar" in s, "children's room": "children" in s,
                "hours": bool(re.search(r"\b(9|nine)\b", s) and re.search(r"\b(6|six|18)\b", s))}
    if name == "vision":
        return {"bar order": re.findall(r"\b(?:red|blue|green)\b", s) == ["blue", "red", "green"]}
    if len(calls) == 1:
        try:
            args = json.loads(calls[0]["arguments"])
        except ValueError:
            args = None
        return {"function": calls[0]["name"] == "find_departures",
                "arguments": args == {"origin": "Dover", "destination": "Ashford", "after": "14:30"}}
    return {"one tool call": False}


def counters(url):
    with urlopen(url + "/metrics", timeout=10) as r:
        text = r.read().decode()
    keys = ["spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total", "prefix_cache_hits_total"]
    return [sum(float(v) for v in re.findall(r"^vllm:" + k + r"\{[^\n]*?\} ([\d.eE+-]+)", text, re.M)) for k in keys]


def measure(url, name, task):
    body = dict(task, model="inkling-2-mini", temperature=0, seed=419, max_tokens=1024, stream=True,
                stream_options={"include_usage": True}, return_token_ids=True, chat_template_kwargs={"reasoning_effort": "none"})
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["cache_salt"] = str(uuid.uuid4())
    req = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    before = counters(url)
    marks, parts, funcs, usage, stop = [], [], {}, {}, None
    start = time.perf_counter()
    with urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                continue
            msg = json.loads(line[6:])
            if msg.get("error"):
                raise RuntimeError(msg["error"])
            usage = msg.get("usage") or usage
            for choice in msg.get("choices", []):
                marks.extend([time.perf_counter()] * len(choice.get("token_ids") or []))
                delta = choice.get("delta") or {}
                parts.append(delta.get("content") or "")
                for call in delta.get("tool_calls") or []:
                    target = funcs.setdefault(call["index"], {"name": "", "arguments": ""})
                    for key in target:
                        target[key] += (call.get("function") or {}).get(key) or ""
                stop = choice.get("finish_reason") or stop
    wall = time.perf_counter() - start
    assert len(marks) == usage["completion_tokens"] and len(marks) > 1
    accepted, proposed, hits = (int(a - b) for a, b in zip(counters(url), before))
    answer, calls = "".join(parts), [funcs[k] for k in sorted(funcs)]
    checks = grade(name, answer, calls)
    checks["completed"] = stop == ("tool_calls" if name == "tool" else "stop")
    return dict(case=name, request_sha256=digest, input=usage["prompt_tokens"], output=len(marks), ttft_s=marks[0]-start,
                wall_s=wall, decode_tps=(len(marks)-1)/(marks[-1]-marks[0]), accepted=accepted, proposed=proposed,
                prefix_hits=hits, answer=answer, calls=calls, checks=checks, passed=all(checks.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("output", type=Path)
    p.add_argument("--bounded-prose", action="store_true", help="Run just the prose fixture with a 256-token thinking budget")
    args = p.parse_args()
    tasks = cases()
    if args.bounded_prose:
        tasks = {"prose": dict(tasks["prose"], thinking_token_budget=256)}
    records = []
    for i in range(4):
        for name, task in tasks.items():
            r = measure(args.url.rstrip("/"), name, task)
            print(i, name, r["passed"], round(r["wall_s"], 3), flush=True)
            if i:
                records.append(dict(trial=i, **r))
                args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
