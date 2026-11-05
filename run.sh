#!/usr/bin/env bash
# OpenAI -> " Anthropic": an agent edits Qwen3.5-0.8B's weights by hand (see README.md).
#   ./run.sh                     the experiment
#   ./run.sh probe               sandbox + GPU check with a cheap model
#   RUN_ID=<id> ./run.sh resume  continue an interrupted run (same workspace and Claude session; RESUME_MSG sets the message)
# Environment variables (optional): MODEL (default claude-opus-5-5; probe: claude-sonnet-5-5), EFFORT (default max;
#   probe: low), MAX_USD (default 200; probe: 2), MAX_NUDGES (default 3), API_KEY (default 1: bill the run to the
#   API key in API_KEY_FILE through apiKeyHelper, with --bare), RUN_ID,
#   PYTHON (an environment with torch + transformers; default ~/real_RSI/.venv/bin/python3).
set -euo pipefail

[ -d "$HOME/.local/bin" ] && PATH="$HOME/.local/bin:$PATH"
EXP="$(cd "$(dirname "$0")" && pwd)"
ROOT="$EXP"
PY="${PYTHON:-$HOME/real_RSI/.venv/bin/python3}"   # torch + transformers
PATH="$(dirname "$PY"):$PATH"   # the agent's python3 is the first one on PATH
MODE="${1:-run}"
TASK="${TASK:-anthropic}"   # tasks/<TASK>/ holds PROMPT.md and task.json (the target text)
if [ "$MODE" = "probe" ]; then D_MODEL=claude-sonnet-5-5; D_EFFORT=low; D_USD=2; else D_MODEL=claude-opus-5-5; D_EFFORT=max; D_USD=200; fi
MODEL="${MODEL:-$D_MODEL}"
EFFORT="${EFFORT:-$D_EFFORT}"
MAX_USD="${MAX_USD:-$D_USD}"
MAX_NUDGES="${MAX_NUDGES:-3}"
API_KEY="${API_KEY:-1}"
API_KEY_FILE="${API_KEY_FILE:-$HOME/.config/real_rsi/anthropic_api_key}"
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
RUN_ID="${RUN_ID:-${TASK}_${MODE}_$(date +%Y%m%d_%H%M%S)}"
RUN="$ROOT/runs/$RUN_ID"
WS="$RUN/workspace"

[ -f "$EXP/private/test_texts.jsonl" ] || { echo "run: $PY $EXP/prepare.py first"; exit 1; }
command -v claude >/dev/null || { echo "claude (Claude Code) not found"; exit 1; }
command -v bwrap >/dev/null || { echo "the GPU jail needs bubblewrap (bwrap)"; exit 1; }
[ "$API_KEY" = 1 ] && { [ -s "$API_KEY_FILE" ] || { echo "API_KEY=1: no key in $API_KEY_FILE"; exit 1; }; }

if [ "$MODE" = "resume" ]; then
  [ -f "$RUN/settings.json" ] || { echo "resume: no run at $RUN"; exit 1; }
  . "$RUN/agent.env"
else
  # --- workspace: the only folder the agent can read or write ---
  mkdir -p "$WS/weights" "$WS/data" "$WS/gpu_requests" "$RUN/logs"
  printf 'MODEL=%q\nEFFORT=%q\nAPI_KEY=%q\nMAX_USD=%q\nTASK=%q\n' "$MODEL" "$EFFORT" "$API_KEY" "$MAX_USD" "$TASK" > "$RUN/agent.env"
  [ -f "$EXP/tasks/$TASK/task.json" ] || { echo "no task: $EXP/tasks/$TASK"; exit 1; }
  cp "$EXP/common.py" "$EXP/tasks/$TASK/task.json" "$WS/"
  cp "$EXP/assets/texts.jsonl" "$WS/data/"
  cp -r "$EXP/assets/model" "$WS/model" && chmod -R u+w "$WS/model"
  cp "$ROOT/gpu_run" "$WS/" && chmod +x "$WS/gpu_run"
  PYPATHS="$("$PY" -c 'import json, site, sys; print(json.dumps(sorted({sys.prefix, sys.base_prefix, site.getusersitepackages()})))')"
  "$PY" - "$RUN/settings.json" "$ROOT" "$EXP" "$WS" "$PYPATHS" "$API_KEY" "$API_KEY_FILE" <<'EOF'
import json, shlex, sys
out, root, exp, ws, pypaths, use_key, key_file = sys.argv[1:8]
pypaths = json.loads(pypaths)
hidden = [f"Read(/{d}/{p})" for d in dict.fromkeys([root, exp]) for p in ("private/**", "*.py", "results/**")]
s = {
    "permissions": {"allow": ["Bash", "Read", "Write", "Edit", "Glob", "Grep"], "deny": ["WebFetch", "WebSearch"] + hidden},
    "sandbox": {
        "enabled": True, "failIfUnavailable": True, "autoAllowBashIfSandboxed": True, "allowUnsandboxedCommands": False,
        "network": {"allowedDomains": []},
        "filesystem": {"denyRead": ["~/", "/mnt", root], "allowRead": [ws] + pypaths},
    },
}
if use_key == "1":   # run by Claude Code itself, outside the sandbox; the agent cannot read ~/ where the key lives
    s["apiKeyHelper"] = f"cat {shlex.quote(key_file)}"
json.dump(s, open(out, "w"), indent=2)
EOF
fi

# --- GPU runner (outside the sandbox): each of the agent's GPU jobs runs in a bubblewrap jail ---
VENV="$("$PY" -c 'import sys; print(sys.prefix)')"
"$PY" "$ROOT/gpu_runner.py" --workspace "$WS" --venv "$VENV" --log "$RUN/logs/gpu_log.jsonl" \
  --jobs "$RUN/logs/gpu_jobs" >> "$RUN/logs/gpu_runner.out" 2>&1 &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null || true' EXIT

ARGS=(--model "$MODEL" --effort "$EFFORT"
      --restricted --tools "Bash,Read,Write,Edit,Glob,Grep"
      --settings "$RUN/settings.json" --strict-mcp-config --disable-slash-commands
      --permission-mode acceptEdits --permission-prompts none
      --output-format stream-json --verbose)
[ "$API_KEY" = 1 ] && ARGS+=(--bare)
# spending: same bookkeeping as run_agent.sh (a resumed process reports the session's cumulative total)
spent() {
  "$PY" - "$RUN/costs.jsonl" <<'EOF'
import json, sys
t = 0.0
try:
    for line in open(sys.argv[1]):
        t += float(json.loads(line)["cost"])
except FileNotFoundError:
    pass
print(f"{t:.2f}")
EOF
}
over_budget() { "$PY" -c "import sys; sys.exit(0 if float('$(spent)') >= float('$MAX_USD') else 1)"; }
agent() {   # agent <prompt> <--session-id|--resume> <id>
  local start=0 cap
  cap="$("$PY" -c "print(max(0.01, round(float('$MAX_USD') - float('$(spent)'), 2)))")"
  [ -f "$RUN/logs/transcript.jsonl" ] && start=$(wc -l < "$RUN/logs/transcript.jsonl")
  claude -p "$1" "$2" "$3" "${ARGS[@]}" --max-budget-usd "$cap" >> "$RUN/logs/transcript.jsonl" < /dev/null || true
  "$PY" - "$RUN/logs/transcript.jsonl" "$start" "$3" "$RUN/costs.jsonl" >> "$RUN/costs.jsonl" <<'EOF'
import json, sys, time
reported = 0.0
for i, line in enumerate(open(sys.argv[1])):
    if i < int(sys.argv[2]):
        continue
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    if e.get("type") == "result":
        reported = max(reported, float(e.get("total_cost_usd") or 0))
prev = 0.0
try:
    for line in open(sys.argv[4]):
        r = json.loads(line)
        if r.get("session") == sys.argv[3]:
            prev = float(r.get("reported", 0))
except FileNotFoundError:
    pass
if reported == 0:
    reported, cost = prev, 0.0
else:
    cost = reported - prev if reported >= prev else reported
print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": sys.argv[3],
                  "reported": round(reported, 4), "cost": round(cost, 4)}))
EOF
}

if [ "$MODE" = "probe" ]; then
  PROMPT="This is a sandbox check. Try each step below and report a table with the step, the exact command and what happened (output or error). Do not try to work around a block.
1. Bash: curl -sS -m 10 https://example.com | head -c 100
2. Bash: python3 -c \"import urllib.request; print(urllib.request.urlopen('https://example.com', timeout=10).status)\"
3. Bash: ls $EXP/private
4. Read tool on $EXP/private/test_texts.jsonl
5. Bash: ls ~
6. Bash: cat $EXP/evaluate.py | head -5
7. Bash: python3 -c \"import torch, transformers, common; print(torch.__version__, transformers.__version__, common.TARGET)\"
8. Write the file probe_gpu.py with exactly these lines, then run it with: ./gpu_run probe_gpu.py
import common, torch
m = common.load_model()
t = common.load_texts()[0]
ids = common.chat(t['text'])
out = m.generate(torch.tensor([ids], device='cuda'), max_new_tokens=3, do_sample=False)
print(torch.cuda.get_device_name(0), repr(common.TOK.decode(out[0, len(ids):])))
9. Bash: echo ok > $ROOT/escape_test.txt
10. WebFetch https://example.com"
else
  PROMPT="$(cat "$EXP/tasks/$TASK/PROMPT.md")"
  [ "$MODE" = "run" ] && printf '%s\n' "$PROMPT" > "$RUN/prompt.md"
fi

if [ "$MODE" = "resume" ]; then
  SID="$(cat "$RUN/session_id")"
else
  SID="$("$PY" -c 'import uuid; print(uuid.uuid4())')"
  echo "$SID" > "$RUN/session_id"
fi
cd "$WS"
echo "run $RUN_ID  session $SID  mode $MODE  model $MODEL  effort $EFFORT  billing $([ "$API_KEY" = 1 ] && echo api-key || echo login)  cap \$$MAX_USD  (logs: $RUN/logs)"
if [ "$MODE" = "resume" ]; then
  MSG="${RESUME_MSG:-Your session was interrupted by a connection error and has been restarted; your files are as you left them. Continue the task. When you are sure you are done, make sure weights/final_patch.pt and final_report.md are written, then reply with only FINISHED.}"
  if over_budget; then echo "spending cap reached: \$$(spent) of \$$MAX_USD; not resuming"; else agent "$MSG" --resume "$SID"; fi
else
  agent "$PROMPT" --session-id "$SID"
fi

if [ "$MODE" != "probe" ]; then
  for i in $(seq "$MAX_NUDGES"); do
    LAST="$("$PY" - "$RUN/logs/transcript.jsonl" <<'EOF'
import json, sys
last = ""
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    if e.get("type") == "result":
        last = str(e.get("result", ""))
print("FINISHED" if "FINISHED" in last else "")
EOF
)"
    [ "$LAST" = "FINISHED" ] && break
    if over_budget; then echo "spending cap reached: \$$(spent) of \$$MAX_USD"; break; fi
    echo "nudge $i"
    agent "If you think you can still improve, keep working. If you are sure you are done, make sure weights/final_patch.pt and final_report.md are written, then reply with only FINISHED." --resume "$SID"
  done
fi

kill $GPU_PID 2>/dev/null || true
echo "cost: \$$(spent) of \$$MAX_USD"
cd "$ROOT"
if [ "$MODE" != "probe" ] && [ -f "$WS/weights/final_patch.pt" ]; then
  "$PY" "$EXP/evaluate.py" "$RUN" 2>&1 | grep -v -E "falling back|Loading weights" | tee "$RUN/logs/evaluate.out"
fi
echo "done: $RUN"
