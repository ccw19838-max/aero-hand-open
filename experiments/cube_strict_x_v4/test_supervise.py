"""Exercise locking, graceful stop, retry accounting and deadline without physics."""
from datetime import datetime, timedelta, timezone
from multiprocessing import Process
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from .supervise import MAX_FAILURES, supervise


def init(root, seconds=30):
    now = datetime.now(timezone.utc)
    clock = {"started_at": now.isoformat(), "deadline_at": (now + timedelta(seconds=seconds)).isoformat()}
    for name in ("clock.json", "job.json"):
        (root / name).write_text(json.dumps(clock))
    return [sys.executable, "-u", "-c"]


def start(root, command):
    proc = Process(target=supervise, args=(root, command), kwargs=dict(poll=.02, grace=.2, retry_delay=.01))
    proc.start()
    return proc


class SupervisorTest(unittest.TestCase):
    def test_persistent_failure_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cmd = init(root) + ["raise SystemExit(2)"]
            self.assertEqual(1, supervise(root, cmd, retry_delay=.001, poll=.01))
            self.assertEqual(1, supervise(root, cmd, retry_delay=.001, poll=.01))
            state = json.loads((root/"supervisor.json").read_text())
            self.assertEqual(MAX_FAILURES, len(state["attempts"]))
            self.assertEqual("retries_exhausted", state["status"])
            self.assertTrue(all((root/entry["log"]).is_file() for entry in state["attempts"]))

    def test_stop_and_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cmd = init(root) + ["import time; time.sleep(30)"]
            proc = start(root, cmd)
            try:
                end = time.monotonic() + 5
                while time.monotonic() < end:
                    p = root/"supervisor.json"
                    if p.exists() and json.loads(p.read_text())["status"] == "running":
                        break
                    time.sleep(.02)
                with self.assertRaises(SystemExit):
                    supervise(root, cmd)
                os.kill(proc.pid, signal.SIGTERM)
                proc.join(2)
                self.assertFalse(proc.is_alive())
                state = json.loads((root/"supervisor.json").read_text())
                self.assertEqual("paused",state["status"])
                self.assertEqual(-signal.SIGTERM,state["attempts"][0]["exit_code"])
            finally:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(2)

    def test_deadline_and_forced_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cmd = init(root, .45) + ["import signal,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);time.sleep(30)"]
            self.assertEqual(0, supervise(root, cmd, grace=.1, poll=.02))
            state = json.loads((root/"supervisor.json").read_text())
            self.assertEqual("deadline_reached", state["status"])
            self.assertEqual(-signal.SIGKILL, state["attempts"][0]["exit_code"])
