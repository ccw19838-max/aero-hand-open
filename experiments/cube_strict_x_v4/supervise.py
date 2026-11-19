"""Run the fixed-deadline experiment and restart its latest committed rollout."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MAX_FAILURES = 4
REPO = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def store(path, state):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temp, path)


def kill_workers(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def supervise(root, command=None, *, grace=30.0, poll=0.4, retry_delay=5.0):
    root = Path(root).resolve()
    job = json.loads((root / "job.json").read_text())
    clock = json.loads((root / "clock.json").read_text())
    if clock["deadline_at"] != job["deadline_at"] or clock["started_at"] != job["started_at"]:
        raise ValueError("Job and original preparation clock disagree")
    deadline = datetime.fromisoformat(clock["deadline_at"]).timestamp()
    path = root / "supervisor.json"
    requested = None

    def request(signum, frame):
        nonlocal requested
        requested = signum

    production = command is None
    if production:
        command = [sys.executable, "-u", "-m", "experiments.cube_strict_x_v4.run", "--root", str(root)]
    with (root / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("The experiment already has a supervisor or a surviving trainer")
        state = json.loads(path.read_text()) if path.exists() else {"attempts": [], "failures": 0}
        if state.get("deadline_at", clock["deadline_at"]) != clock["deadline_at"]:
            raise ValueError("Original supervisor deadline changed")
        if state.get("status") == "completed":
            return 0
        state.update(pid=os.getpid(), trainer_pid=None, deadline_at=clock["deadline_at"],
                     supervisor_started_at=now(), status="initializing")
        store(path, state)
        logdir = root / "logs"
        logdir.mkdir(exist_ok=True)
        old = {s: signal.signal(s, request) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            while not requested and time.time() < deadline and state["failures"] < MAX_FAILURES:
                number = len(state["attempts"]) + 1
                name = f"logs/trainer_{number:03d}.log"
                with (root / name).open("a", buffering=1) as log:
                    # The child holds the lock too, so a surviving trainer cannot
                    # be duplicated even while it is still importing its modules.
                    child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                             pass_fds=(lock.fileno(),))
                    entry = dict(attempt=number, trainer_pid=child.pid, log=name, started_at=now())
                    state["attempts"].append(entry)
                    state.update(trainer_pid=child.pid, status="running")
                    store(path, state)
                    sent = None
                    while child.poll() is None:
                        if requested or time.time() >= deadline:
                            if sent is None:
                                child.send_signal(requested or signal.SIGTERM)
                                sent = time.monotonic()
                                state["status"] = "saving_at_cutoff" if not requested else "pausing"
                                store(path, state)
                            elif time.monotonic() - sent >= grace:
                                kill_workers(child.pid)
                        time.sleep(poll)
                    kill_workers(child.pid)
                    entry.update(exit_code=child.returncode, finished_at=now())
                    state.update(trainer_pid=None)
                if requested or time.time() >= deadline:
                    break
                finished = child.returncode == 0
                if finished and production:
                    progress = root / "state.json"
                    finished = progress.exists() and json.loads(progress.read_text()).get("phase") == "done"
                    if not finished:
                        entry["unexpected_exit"] = "Trainer exited before completing final delivery"
                if finished:
                    state.update(status="completed", finished_at=now())
                    store(path, state)
                    return 0
                state["failures"] += 1
                state["status"] = "retry_wait"
                store(path, state)
                retry_at = min(time.time() + retry_delay, deadline)
                while not requested and state["failures"] < MAX_FAILURES and time.time() < retry_at:
                    time.sleep(min(poll, max(0., retry_at - time.time())))
            state["status"] = "paused" if requested else (
                "deadline_reached" if time.time() >= deadline else "retries_exhausted")
            state.update(trainer_pid=None, finished_at=now())
            store(path, state)
            return int(state["status"] == "retries_exhausted")
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    raise SystemExit(supervise(parser.parse_args().root))
