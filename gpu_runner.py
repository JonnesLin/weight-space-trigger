"""Runs the agent's Python scripts on the GPU, one job at a time. Started by run.sh.

Claude Code's sandbox cannot pass a GPU through (it builds a fresh /dev), so this runner lives outside it. It never runs
the agent's code directly: every job runs in a bubblewrap jail of our own with the sandbox's restrictions, plus the GPU:
  - no network (all namespaces unshared);
  - the filesystem read-only, with /home, /mnt and /tmp replaced by empty in-memory folders, and only the workspace
    (read-write) and the Python environment (read-only) mounted back;
  - a fresh /dev with only the GPU device nodes added;
  - a clean environment (nothing inherited from this process).

Protocol (files in <workspace>/gpu_requests/):
  the agent's gpu_run writes  job_<id>.json  {"script": "<absolute path>", "args": [...], "timeout": seconds}
  this runner writes          out_<id>.json  {"returncode", "stdout", "stderr", "seconds"} (plus "error" on failure)
Every job is logged (script sha256, arguments, return code, time), and a copy of each script is kept outside the
workspace for the audit.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

DEFAULT_TIMEOUT, MAX_TIMEOUT = 600, 3600
MAX_OUTPUT = 100_000   # characters kept from the end of stdout and of stderr
GPU_DEVICES = ["/dev/dxg", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"]


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def tail(s):
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    return s[-MAX_OUTPUT:]


def jail(ws, venv, writable=True, binds=()):
    """The bubblewrap command a job runs under (WSL2 GPUs: /dev/dxg; native Linux: the /dev/nvidia* nodes).
    writable=False mounts the workspace read-only (scoring the agent's own model); binds adds (flag, source, target)
    mounts, with targets under /tmp."""
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/", "/", "--tmpfs", "/home", "--tmpfs", "/mnt", "--tmpfs", "/tmp",
           "--dev", "/dev", "--tmpfs", "/dev/shm", "--proc", "/proc"]
    for d in GPU_DEVICES + sorted(glob.glob("/dev/nvidia[0-9]*")):
        if os.path.exists(d):
            cmd += ["--dev-bind", d, d]
    cmd += ["--ro-bind", venv, venv, "--bind" if writable else "--ro-bind", ws, ws]
    for flag, src, dst in binds:
        cmd += [flag, src, dst]
    return cmd + ["--clearenv", "--setenv", "PATH", f"{venv}/bin:/usr/bin:/bin", "--setenv", "HOME", "/tmp",
                  "--setenv", "PYTHONUNBUFFERED", "1", "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--chdir", ws]


def run_job(req, ws, venv, jobs):
    """Validate and run one request. Returns (output for the agent, log fields)."""
    r = json.loads(req.read_text())
    script = Path(r["script"]).resolve()
    # compare by file identity: one folder can have two path spellings
    if not script.is_file() or not any(p.exists() and p.samefile(ws) for p in script.parents):
        raise ValueError("the script must be a file inside the workspace")
    argv = [str(a) for a in r.get("args") or []]
    timeout = max(1, min(int(r.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT))
    data = script.read_bytes()
    log = {"script": os.path.relpath(script, ws), "sha256": hashlib.sha256(data).hexdigest(), "args": argv,
           "timeout": timeout}
    (jobs / f"{req.stem[4:]}_{script.name}").write_bytes(data)
    t0 = time.time()
    try:
        p = subprocess.run(jail(str(ws), str(venv)) + [str(venv / "bin" / "python3"), str(script), *argv],
                           capture_output=True, timeout=timeout)
        out = {"returncode": p.returncode, "stdout": tail(p.stdout), "stderr": tail(p.stderr)}
    except subprocess.TimeoutExpired as e:
        out = {"returncode": None, "error": f"timed out after {timeout} s and was stopped",
               "stdout": tail(e.stdout), "stderr": tail(e.stderr)}
    out["seconds"] = round(time.time() - t0, 1)
    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--venv", required=True, help="the Python environment jobs use (mounted read-only)")
    ap.add_argument("--log", required=True)
    ap.add_argument("--jobs", required=True, help="folder outside the workspace for copies of the scripts")
    args = ap.parse_args()
    ws, venv = Path(args.workspace).resolve(), Path(args.venv).resolve()
    reqdir = ws / "gpu_requests"
    reqdir.mkdir(exist_ok=True)
    jobs = Path(args.jobs)
    jobs.mkdir(parents=True, exist_ok=True)

    while True:
        for req in sorted(reqdir.glob("job_*.json"), key=lambda p: p.stat().st_mtime):
            res = reqdir / req.name.replace("job_", "out_", 1)
            if res.exists():
                continue
            entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "job": req.stem[4:]}
            try:
                out, fields = run_job(req, ws, venv, jobs)
                entry.update(fields)
            except Exception as e:  # reported to the agent
                out = {"returncode": None, "error": f"{type(e).__name__}: {e}"}
            write_json(res, out)
            entry["result"] = {k: out.get(k) for k in ("returncode", "seconds", "error")}
            with open(args.log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
