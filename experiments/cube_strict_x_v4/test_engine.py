"""Run the real four-worker preparation smoke test in a disposable experiment."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aero-cube-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aero-cube-cache")
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import random
import shutil
import tempfile
import time

import numpy as np
import torch
from experiments.cube_adaptive_60h.storage import atomic_json
from .engine import BATCH_STEPS, Engine, load_for_eval


def weights(model):
    checksum = hashlib.sha256()
    for name, tensor in model.policy.state_dict().items():
        checksum.update(name.encode())
        checksum.update(tensor.detach().cpu().numpy().tobytes())
    return checksum.hexdigest()


def check(root):
    root = Path(root).resolve()
    job = json.loads((root / "job.json").read_text())
    report = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                  original_deadline_at=job["deadline_at"])
    atomic_json(root / "engine_checks.json", report)
    with tempfile.TemporaryDirectory(prefix="cube-x-engine-") as temp:
        smoke = Path(temp)
        shutil.copy2(root / "job.json", smoke / "job.json")
        shutil.copytree(root / "inputs", smoke / "inputs")
        with Engine(smoke, "v3") as first:
            obs = first.model._last_obs.copy()
            expected = first.model.predict(obs, deterministic=True)[0]
            std = first.model.policy.log_std.detach().clone()
            baseline_checkpoint = first.save("validation_initial")
            assert first.steps == first.epochs == 0 and len(first.worker_pids) == 4
        with Engine(smoke, "v4") as engine:
            raw = engine.model._last_obs
            np.testing.assert_array_equal(raw[:, :143], obs)
            np.testing.assert_array_equal(engine.model.predict(raw, deterministic=True)[0], expected)
            torch.testing.assert_close(engine.model.policy.log_std, std, rtol=0, atol=0)
            baseline, _ = load_for_eval(baseline_checkpoint)
            # A zeroed action head alone would trivially pass. Check nonzero
            # hidden activations and arbitrary nonzero warning channels too.
            sample = np.random.default_rng(23).normal(size=(13,149)).astype(np.float32)
            with torch.no_grad():
                x = baseline.policy.features_extractor(torch.from_numpy(sample[:,:143]))
                y = engine.model.policy.features_extractor(torch.from_numpy(sample))
                old_pi, old_v = baseline.policy.mlp_extractor(x)
                new_pi, new_v = engine.model.policy.mlp_extractor(y)
                torch.testing.assert_close(old_pi,new_pi,rtol=1e-5,atol=2e-6)
                torch.testing.assert_close(old_v,new_v,rtol=1e-5,atol=2e-6)
            t = time.monotonic()
            initial = weights(engine.model)
            assert engine.advance() and engine.steps == BATCH_STEPS and engine.epochs == 4
            checkpoint = engine.save("validation_first_update")
            assert initial != weights(engine.model)
            first_time = time.monotonic() - t
            deterministic = engine.model.predict(raw,deterministic=True)[0]
            expected_rng = random.random(), np.random.uniform(), torch.rand(1).item()
            expected_obs = engine.model._last_obs.copy()
            saved_steps = {int(v["step"]) for v in engine.model.policy.optimizer.state.values()}
            saved_rms = (engine.normalizer.ret_rms.mean.copy(), engine.normalizer.ret_rms.var.copy())
        with Engine(smoke, "v4", checkpoint=checkpoint) as engine:
            assert (random.random(), np.random.uniform(), torch.rand(1).item()) == expected_rng
            np.testing.assert_array_equal(engine.model._last_obs, expected_obs)
            np.testing.assert_array_equal(engine.model.predict(raw,deterministic=True)[0],deterministic)
            np.testing.assert_array_equal(engine.normalizer.ret_rms.mean,saved_rms[0])
            np.testing.assert_array_equal(engine.normalizer.ret_rms.var,saved_rms[1])
            assert {int(v["step"]) for v in engine.model.policy.optimizer.state.values()} == saved_steps
            assert engine.deadline == datetime.fromisoformat(job["deadline_at"]).timestamp()
            t = time.monotonic()
            assert engine.advance() and engine.steps == 2 * BATCH_STEPS and engine.epochs == 8
            after_optimizer = {int(v["step"]) for v in engine.model.policy.optimizer.state.values()}
            assert min(after_optimizer) > max(saved_steps)
            # Exercise the partial-rollout cutoff, then prove its cursor can be loaded.
            assert not engine.advance(deadline=time.time()+.3)
            assert engine.steps == 2 * BATCH_STEPS and engine.epochs == 8
            second = engine.save("validation_interrupted_rollout")
            second_time = time.monotonic() - t
        with Engine(smoke, "v4", checkpoint=second) as engine:
            assert engine.steps == 2 * BATCH_STEPS and engine.epochs == 8
            engine.set_stage(10.)
            _, meta = load_for_eval(engine.save("validation_transition"))
            assert meta["stage"] == 10. and meta["deadline_at"] == job["deadline_at"]
        report.update(status="passed", completed_at=datetime.now(timezone.utc).isoformat(),
                      paired_initial_action_std_and_features=True, trained_steps_before_resume=BATCH_STEPS,
                      trained_steps_after_resume=2*BATCH_STEPS, epochs_before_after=[4,8],
                      optimizer_steps_before_after=[max(saved_steps), min(after_optimizer)],
                      rng_exact=True, worker_replays_and_observations_exact=True, reward_statistics_exact=True,
                      partial_rollout_discarded=True, original_deadline_restored=True, curriculum_transition=True,
                      worker_physical_steps_ms=[1,2,1,2], seconds_per_update=[round(first_time,2),round(second_time,2)])
    atomic_json(root / "engine_checks.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True)
    args=p.parse_args()
    try:
        check(args.root)
    except BaseException as exc:
        record=Path(args.root)/"engine_checks.json"
        data=json.loads(record.read_text()) if record.exists() else {}
        data.update(status="failed", error=repr(exc));atomic_json(record,data)
        raise
