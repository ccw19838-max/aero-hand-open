"""Same 20-degree palm-up physics; a separately versioned envelope curriculum.

No mass, size, friction, force, or hand-geometry changes are used to manufacture
five-finger contact. The new reset must pass a free-body preparation check.
"""
from experiments.cube_reorient.model import (
    scene_and_assets, build_model, model_fingerprint, model_ids,
    native_contacts, native_cube_velocity, fixed_axis_basis,
    HOME_QPOS, HOME_CTRL,
)
