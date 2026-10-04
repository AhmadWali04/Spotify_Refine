"""Runs the pipeline steps (pull, tags, sort) for the web app as background subprocesses.

Each step is the same `python -m ...` command the README documents, so the web app and the CLI
can never drift apart. Output lines are kept for the progress view; "123/456" in a line becomes
a progress fraction.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from src import paths

DEMO_LIBRARY = paths.RAW / "library_synthetic.json"
TAGS_ONLY_CONFIG = "configs/e1b_tags_lsa.yaml"     # used when Phase 1 hasn't picked a featurizer yet
STEPS = ("pull", "tags", "sort")
PROGRESS = re.compile(r"(\d+)\s*/\s*(\d+)")
KEEP_LINES = 200


def commands(lib_path: Path) -> dict[str, list[str] | None]:
    py = [sys.executable, "-u", "-m"]
    if lib_path.resolve() == DEMO_LIBRARY.resolve():
        return {"pull": py + ["src.fetch.synthetic"], "tags": None,           # synthetic writes its own tags
                "sort": py + ["src.sorter.run", "--library", str(lib_path)]}
    sort = py + ["src.sorter.run"]
    if not paths.for_library(lib_path, "decision.json").exists():
        sort += ["--config", TAGS_ONLY_CONFIG]
    return {"pull": py + ["src.fetch.spotify"], "tags": py + ["src.fetch.lastfm"], "sort": sort}


def _pulled_at(lib_path: Path) -> str | None:
    try:
        return json.loads(lib_path.read_text()).get("pulled_at") if lib_path.exists() else None
    except json.JSONDecodeError:
        return None


def done_state(lib_path: Path) -> dict[str, bool]:
    """Which steps already have up-to-date output on disk."""
    pulled = _pulled_at(lib_path)
    sugg = paths.for_library(lib_path, "suggestions.json")
    sorted_ok = False
    if pulled and sugg.exists():
        try:
            sorted_ok = json.loads(sugg.read_text()).get("pulled_at") == pulled
        except json.JSONDecodeError:
            pass
    return {"pull": pulled is not None, "tags": paths.tags_path(lib_path).exists(), "sort": sorted_ok}


class Jobs:
    def __init__(self, on_step_done=None):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None
        self.on_step_done = on_step_done
        self.steps: dict[str, dict] = {s: self._blank() for s in STEPS}

    @staticmethod
    def _blank() -> dict:
        return {"status": "idle", "lines": [], "progress": None, "started": None, "ended": None}

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def snapshot(self) -> dict:
        with self.lock:
            return {"running": self.running,
                    "steps": {k: {**v, "lines": v["lines"][-12:]} for k, v in self.steps.items()}}

    def start(self, lib_path: Path, steps: list[str]) -> None:
        bad = [s for s in steps if s not in STEPS]
        if bad:
            raise ValueError(f"Unknown step(s): {', '.join(bad)}")
        if self.running:
            raise RuntimeError("A step is already running. Wait for it to finish.")
        cmds = commands(lib_path)
        with self.lock:
            for s in steps:
                self.steps[s] = {**self._blank(), "status": "queued" if cmds[s] else "skipped"}
        self.thread = threading.Thread(target=self._run, args=(lib_path, steps, cmds), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _run(self, lib_path: Path, steps: list[str], cmds: dict) -> None:
        for name in steps:
            if not cmds[name]:
                continue
            st = self.steps[name]
            with self.lock:
                st.update(status="running", started=time.time())
            code = self._exec(cmds[name], st)
            with self.lock:
                st.update(status="done" if code == 0 else "failed", ended=time.time())
                if code == 0:
                    st["progress"] = 1.0
            if code != 0:
                for rest in steps[steps.index(name) + 1:]:
                    with self.lock:
                        if self.steps[rest]["status"] == "queued":
                            self.steps[rest]["status"] = "idle"
                return
            if self.on_step_done:
                self.on_step_done(name)

    def _exec(self, cmd: list[str], st: dict) -> int:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
        try:
            self.proc = subprocess.Popen(cmd, cwd=paths.ROOT, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as e:
            with self.lock:
                st["lines"].append(str(e))
            return 1
        for line in self.proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            with self.lock:
                st["lines"] = (st["lines"] + [line])[-KEEP_LINES:]
                m = PROGRESS.search(line)
                if m and int(m.group(2)) > 0:
                    st["progress"] = min(1.0, int(m.group(1)) / int(m.group(2)))
        return self.proc.wait()
