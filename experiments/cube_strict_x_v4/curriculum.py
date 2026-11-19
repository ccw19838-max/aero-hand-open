"""Revise this live experiment at a committed PPO boundary while retaining its history."""
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.cube_adaptive_60h.storage import atomic_json, digest

ORIGINAL_DEGREES = (5, 10, 20, 30, 45, 60, 75, 90)
REVISED_DEGREES = (30, 60, 90)


def read(path):
    return json.loads(path.read_text())


def record_amendment(root, state, job):
    """Called under the experiment lock. A repeat invocation reuses the first boundary."""
    manifest = root / "amendment.json"
    if manifest.exists():
        return
    if (state["phase"] != "curriculum" or state.get("final_quorum") or
            (root/"final"/"frozen.json").exists()):
        raise ValueError("Changing the course requires an active V4 curriculum before freezing")
    checkpoint = (root/"arms"/"v4"/"latest").resolve(strict=True)
    meta = read(checkpoint/"metadata.json")
    if (Path(state["last_v4_checkpoint"]).resolve() != checkpoint or
            meta["arm"] != "v4" or meta["stage"] != state["stage"] or
            meta["training_steps"] != state["steps"]["v4"] or
            meta["job_sha256"] != state["job_sha256"]):
        raise ValueError("Save a complete V4 rollout before changing the curriculum")
    archive = root/"state_before_30_60_90.json"
    atomic_json(archive, state)
    atomic_json(manifest, dict(
        created_at=datetime.now(timezone.utc).isoformat(), job_sha256=state["job_sha256"],
        deadline_at=job["deadline_at"], curriculum_degrees=list(REVISED_DEGREES),
        previous_curriculum_degrees=list(ORIGINAL_DEGREES), previous_stage=state["stage"],
        checkpoint=str(checkpoint), policy_sha256=digest(checkpoint/"policy.zip"),
        training_steps=meta["training_steps"], ppo_training_epochs=meta["ppo_training_epochs"],
        previous_state=archive.name, previous_state_sha256=digest(archive)))


def apply_amendment(root, state, job):
    """Project the immutable amendment onto scheduler state, including after a restart."""
    manifest = root/"amendment.json"
    if not manifest.exists():
        if "amendment_sha256" in state:
            raise ValueError("Missing curriculum amendment")
        return
    amendment = read(manifest)
    signature = digest(manifest)
    if (amendment["job_sha256"] != state["job_sha256"] or amendment["deadline_at"] != job["deadline_at"]
            or amendment["curriculum_degrees"] != list(REVISED_DEGREES)):
        raise ValueError("Curriculum amendment does not match the experiment")
    archive = root/amendment["previous_state"]
    checkpoint = Path(amendment["checkpoint"])
    meta = read(checkpoint/"metadata.json")
    if (digest(archive) != amendment["previous_state_sha256"] or
            digest(checkpoint/"policy.zip") != amendment["policy_sha256"] or
            meta["job_sha256"] != state["job_sha256"] or meta["arm"] != "v4" or
            meta["stage"] != amendment["previous_stage"] or
            meta["training_steps"] != amendment["training_steps"] or
            meta["ppo_training_epochs"] != amendment["ppo_training_epochs"]):
        raise ValueError("Archived curriculum boundary failed its integrity check")
    if "amendment_sha256" in state:
        if state["amendment_sha256"] != signature or state["curriculum_degrees"] != list(REVISED_DEGREES):
            raise ValueError("Curriculum changed after training resumed")
        return
    if read(archive) != state or (root/"arms"/"v4"/"latest").resolve(strict=True) != checkpoint:
        raise ValueError("Curriculum amendment and saved training boundary disagree")
    state.update(curriculum_degrees=list(REVISED_DEGREES), stage=REVISED_DEGREES[0], streak=0,
                 final_quorum=False, qualified={}, history={}, best={}, practice={},
                 last_evaluation=None, amendment_sha256=signature)
