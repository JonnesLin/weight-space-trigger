#!/usr/bin/env python3
"""Follow an agent run live: what the agent says, the commands it runs, their output, GPU jobs.

    python3 watch.py               # newest run under runs/
    python3 watch.py <run_dir>     # a specific run
    python3 watch.py --full        # do not shorten tool output
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
FULL = "--full" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT_LINES = 10 ** 9 if FULL else 12

C = {"say": "\033[96m", "cmd": "\033[93m", "out": "\033[90m", "err": "\033[91m",
     "score": "\033[92;1m", "meta": "\033[95m", "think": "\033[2;3m", "end": "\033[0m"}


def paint(kind, text):
    return f"{C[kind]}{text}{C['end']}"


def short(text, n=OUT_LINES):
    lines = str(text).rstrip().splitlines()
    if len(lines) > n:
        lines = lines[:n] + [f"... ({len(lines) - n} more lines)"]
    return "\n".join("    " + l[:400] for l in lines)


def result_text(content):
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def show(e, live):
    stamp = datetime.now().strftime("%H:%M:%S") + " " if live else ""
    t = e.get("type")
    if t == "system" and e.get("subtype") == "init":
        print(paint("meta", f"=== session start  model={e.get('model')}  tools={e.get('tools')}"))
    elif t == "assistant":
        for p in e["message"].get("content", []):
            k = p.get("type")
            if k == "text" and p["text"].strip():
                print(stamp + paint("say", "AI: ") + p["text"].strip())
            elif k == "thinking" and p.get("thinking", "").strip():
                print(stamp + paint("think", "(thinking) " + p["thinking"].strip()[:600]))
            elif k == "tool_use":
                i = p.get("input", {})
                if p["name"] == "Bash":
                    desc = f"   # {i['description']}" if i.get("description") else ""
                    style = "score" if "score_b" in i.get("command", "") else "cmd"
                    print(stamp + paint(style, "$ " + i.get("command", "").strip()) + paint("out", desc))
                elif p["name"] in ("Write", "Edit"):
                    body = i.get("content") or i.get("new_string") or ""
                    print(stamp + paint("cmd", f"[{p['name']}] {i.get('file_path')}"))
                    print(paint("out", short(body, 8)))
                else:
                    print(stamp + paint("cmd", f"[{p['name']}] {json.dumps(i, ensure_ascii=False)[:300]}"))
    elif t == "user":
        content = e.get("message", {}).get("content")
        if isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") == "tool_result":
                    text = result_text(p.get("content"))
                    style = "err" if p.get("is_error") else ("score" if '"left"' in text else "out")
                    print(paint(style, short(text)))
        elif isinstance(content, str):
            print(stamp + paint("meta", "PROMPT: " + content[:300]))
    elif t == "result":
        print(paint("meta", f"=== session end  {e.get('subtype')}  turns={e.get('num_turns')}  "
                            f"{(e.get('duration_ms') or 0) / 60000:.1f} min  cost≈${e.get('total_cost_usd') or 0:.2f}"))
    sys.stdout.flush()


def main():
    if args:
        run = Path(args[0]).expanduser()
    else:
        runs = sorted((d for d in RUNS.iterdir() if (d / "logs" / "transcript.jsonl").exists()),
                      key=lambda d: (d / "logs" / "transcript.jsonl").stat().st_mtime)
        if not runs:
            sys.exit(f"no runs under {RUNS}")
        run = runs[-1]
    path = run / "logs" / "transcript.jsonl"
    state = run / "score_state.json"
    print(paint("meta", f"following {path}  (Ctrl+C to stop)"))
    with open(path) as f:
        buf, live, last_beat = "", False, time.time()
        while True:
            chunk = f.readline()
            if not chunk:
                if not live:
                    live = True
                    print(paint("meta", "--- caught up, now live ---"))
                if time.time() - last_beat > 120:  # heartbeat so a quiet stretch is not mistaken for a hang
                    left = json.loads(state.read_text())["left"] if state.exists() else "?"
                    print(paint("meta", f"{datetime.now():%H:%M:%S} ... still running, score_b left: {left}"))
                    last_beat = time.time()
                time.sleep(0.5)
                continue
            buf += chunk
            if not buf.endswith("\n"):
                continue
            line, buf = buf, ""
            last_beat = time.time()
            try:
                show(json.loads(line), live)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
