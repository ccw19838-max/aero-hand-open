"""Model gates and bounded, unassisted static-grasp search.

Only a fully passing report writes reference.npz. A failed best_candidate.npz is
diagnostic evidence and must never be used to bypass the training gate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.stats import qmc

from .config import PenSpinConfig
from .model import (
    PEN_COM, PEN_INERTIA, PEN_LENGTH, PEN_MASS, PEN_RADIUS,
    build_model, model_fingerprint, model_ids, native_contacts,
    native_pen_velocity, task_fingerprint, thumb_frame,
)


def stable_sample(model, data, ids, config):
    contacts = native_contacts(model, data, ids)
    linear, angular = native_pen_velocity(model, data, ids)
    fingers = contacts["finger_forces"] >= config.contact_force
    valid = bool(
        fingers[0] and np.any(fingers[1:])
        and np.linalg.norm(linear) < config.grip_velocity
        and np.linalg.norm(angular) < config.grip_omega
        and not contacts["invalid_support"]
        and contacts["max_penetration"] <= config.max_penetration
        and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    )
    return valid, {
        **contacts,
        "linear_speed": float(np.linalg.norm(linear)),
        "angular_speed": float(np.linalg.norm(angular)),
    }


def _new_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def reference_geometry(model, data, ids, reference, config):
    """Check the exact frozen-axis envelope used by trajectory judgment."""
    n0, c0 = np.asarray(reference["n0"]), np.asarray(reference["c0"])
    finite = np.isfinite(n0).all() and np.isfinite(c0).all()
    axis_norm = np.linalg.norm(n0)
    if not finite or axis_norm < 1e-8:
        return {"passed": False, "reason": "nonfinite_or_zero_reference_axis"}
    n0 = n0 / axis_norm
    center = data.site_xpos[ids["site_center"]]
    endpoint = data.site_xpos[ids["site_a"]]
    direction = endpoint - data.site_xpos[ids["site_b"]]
    direction /= np.linalg.norm(direction)
    relative, endpoint_relative = center - c0, endpoint - c0
    projected_direction = direction - np.dot(direction, n0) * n0
    projected_endpoint = endpoint_relative - np.dot(endpoint_relative, n0) * n0
    thumb_center, thumb_axis = thumb_frame(model, data, ids)
    thumb_relative = thumb_center - c0
    measurements = {
        "plane_degrees": float(np.degrees(np.arcsin(np.clip(abs(np.dot(direction, n0)), 0, 1)))),
        "center_distance_m": float(np.linalg.norm(relative)),
        "axial_distance_m": float(abs(np.dot(relative, n0))),
        "endpoint_radius_m": float(np.linalg.norm(projected_endpoint)),
        "direction_projection": float(np.linalg.norm(projected_direction)),
        "pivot_drift_m": float(np.linalg.norm(thumb_relative - np.dot(thumb_relative, n0) * n0)),
        "thumb_axis_degrees": float(np.degrees(np.arccos(np.clip(np.dot(thumb_axis, n0), -1, 1)))),
    }
    checks = {
        "finite": bool(np.isfinite(list(measurements.values())).all()),
        "plane": measurements["plane_degrees"] <= config.max_plane_degrees,
        "center_distance": measurements["center_distance_m"] <= config.max_center_distance,
        "axial_distance": measurements["axial_distance_m"] <= config.max_axial_distance,
        "endpoint_radius": measurements["endpoint_radius_m"] >= config.min_endpoint_radius,
        "direction_projection": measurements["direction_projection"] > 1e-6,
        "pivot_drift": measurements["pivot_drift_m"] <= config.max_pivot_drift,
        "thumb_axis": measurements["thumb_axis_degrees"] <= config.max_thumb_axis_degrees,
    }
    return {"passed": bool(all(checks.values())), "checks": checks, **measurements}


def check_reference_geometry(reference, config):
    model = build_model(config); ids = model_ids(model); data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"]; data.qvel[:] = reference["qvel"]
    data.ctrl[:] = reference["ctrl"]; mujoco.mj_forward(model, data)
    return reference_geometry(model, data, ids, reference, config)


def check_model(model):
    ids = model_ids(model)
    body = ids["pen_body"]
    failures = []
    checks = {
        "total_mass": bool(np.isclose(model.body_mass[body], PEN_MASS, atol=1e-12)),
        "center_of_mass": bool(np.allclose(model.body_ipos[body], PEN_COM, atol=1e-12)),
        "inertia": bool(np.allclose(model.body_inertia[body], PEN_INERTIA, atol=1e-12)),
        "capsule_length": bool(np.isclose(2 * model.geom_size[ids["pen_geom"], :2].sum(), PEN_LENGTH)),
        "radius": bool(np.isclose(model.geom_size[ids["pen_geom"], 0], PEN_RADIUS)),
        "gravity": bool(np.allclose(model.opt.gravity, [0, 0, -9.81])),
        "seven_actuators": model.nu == 7,
    }
    visual = [model.geom(n).id for n in
              ("pen_middle_visual", "pen_heavy_visual", "pen_light_visual")]
    checks["visual_noncolliding"] = all(
        model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 for g in visual
    )
    pen = ids["pen_geom"]
    finger_collision = np.flatnonzero((ids["geom_finger"] >= 0) & (model.geom_contype != 0))
    checks["all_fingers_collision_enabled"] = bool(len(finger_collision) and all(
        (int(model.geom_contype[pen]) & int(model.geom_conaffinity[g]))
        or (int(model.geom_contype[g]) & int(model.geom_conaffinity[pen]))
        for g in finger_collision
    ))
    checks["specified_contact_parameters"] = bool(
        model.geom_condim[pen] == 3
        and np.isclose(model.geom_friction[pen, 0], .6)
        and all(model.geom_priority[pen] > model.geom_priority[g] for g in finger_collision)
    )
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    return {"passed": not failures, "checks": checks, "failures": failures}


def check_freeflight(config):
    """Compare 2ms/1ms independent integrations in a contact-free pen region."""
    positions, velocities = [], []
    for dt in (config.sim_dt, config.sim_dt / 2):
        m = build_model(replace(config, sim_dt=dt))
        d = _new_data(m)
        ids = model_ids(m)
        for _ in range(round(.1 / dt)):
            mujoco.mj_step(m, d)
        positions.append(d.qpos[ids["pen_qpos_adr"]:ids["pen_qpos_adr"] + 3].copy())
        velocities.append(d.qvel[ids["pen_dof_adr"]:ids["pen_dof_adr"] + 6].copy())
    error = float(np.linalg.norm(positions[0] - positions[1]))
    velocity_error = float(np.linalg.norm(velocities[0] - velocities[1]))
    passed = bool(np.isfinite(positions).all() and error < .001 and velocity_error < 1e-5)
    return {"passed": passed, "duration_seconds": .1, "timesteps": [config.sim_dt, config.sim_dt / 2],
            "position_difference_m": error, "velocity_difference": velocity_error,
            "position_tolerance_m": .001, "velocity_tolerance": 1e-5}


def check_mjx_contact(model):
    """Require actual Aero thumb contact, plus an isolated collision-pair check."""
    import jax
    from mujoco import mjx
    from mujoco.mjx._src import support
    # A tiny test scene alone cannot catch disabled masks in the actual hand.
    # Deliberately place the capsule 1mm into a real distal-thumb collision box;
    # this is a collision diagnostic, never a candidate grasp or reference.
    ids = model_ids(model)
    d_full = _new_data(model)
    thumb = model.geom("th_tip").id
    rotation = d_full.geom_xmat[thumb].reshape(3, 3)
    qadr = ids["pen_qpos_adr"]
    d_full.qpos[qadr:qadr+3] = (d_full.geom_xpos[thumb]
        + rotation[:, 0] * (model.geom_size[thumb, 0] + PEN_RADIUS - .001))
    d_full.qpos[qadr+3:qadr+7] = _quat_z_to(rotation[:, 2])
    d_full.qvel[ids["pen_dof_adr"]:ids["pen_dof_adr"]+6] = 0
    mujoco.mj_forward(model, d_full)
    full_native = native_contacts(model, d_full, ids)
    actual_contact_dimensions, actual_contact_frictions = [], []
    for contact in d_full.contact:
        if ids["pen_geom"] in (contact.geom1, contact.geom2):
            actual_contact_dimensions.append(int(contact.dim))
            actual_contact_frictions.append(float(contact.friction[0]))
    mm_full = mjx.put_model(model)
    md_full = jax.jit(mjx.forward)(mm_full, mjx.put_data(model, d_full))
    md_full.qpos.block_until_ready()
    full_force, full_indices = support.contact_force_dim(mm_full, md_full, 3)
    geoms = np.asarray(md_full._impl.contact.geom)[np.asarray(full_indices)]
    is_pen = (geoms[:, 0] == ids["pen_geom"]) | (geoms[:, 1] == ids["pen_geom"])
    others = np.where(geoms[:, 0] == ids["pen_geom"], geoms[:, 1], geoms[:, 0])
    thumb_contacts = is_pen & (ids["geom_finger"][np.clip(others, 0, model.ngeom-1)] == 0)
    full_mjx_thumb_force = float(np.maximum(np.asarray(full_force)[:, 0], 0)[thumb_contacts].sum())
    mjx_thumb_friction = np.asarray(md_full._impl.contact.friction)[np.asarray(full_indices)][thumb_contacts, 0]
    actual_hand_passed = bool(full_native["finger_forces"][0] > 0
                             and full_mjx_thumb_force > 0
                             and actual_contact_dimensions
                             and all(v == 3 for v in actual_contact_dimensions)
                             and np.allclose(actual_contact_frictions, .6)
                             and np.allclose(mjx_thumb_friction, .6)
                             and np.isfinite(np.asarray(md_full.qpos)).all())
    xml = """<mujoco><option timestep=".002"/><worldbody>
      <geom type="box" size=".08 .03 .01"/>
      <body pos="0 0 .014"><freejoint/>
      <inertial pos="0 0 .006875" mass=".016"
        diaginertia="3.874375e-5 3.874375e-5 2e-7"/>
      <geom type="capsule" size=".005 .060"
        quat=".7071067811865476 0 .7071067811865476 0"
        friction=".6 .001 .0001" condim="3"/>
      </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    native_count = d.ncon
    native_force = np.zeros(6)
    if native_count:
        mujoco.mj_contactForce(m, d, 0, native_force)
    mm = mjx.put_model(m)
    md = mjx.put_data(m, d)
    md = jax.jit(mjx.forward)(mm, md)
    md.qpos.block_until_ready()
    active = np.asarray(md._impl.contact.dist) < 0
    force_sum = float(np.abs(np.asarray(md._impl.efc_force)).sum())
    passed = bool(actual_hand_passed and native_count > 0 and active.any() and force_sum > 0
                  and np.isfinite(np.asarray(md.qpos)).all())
    return {"passed": passed, "full_aero_model_converted": True,
            "actual_aero_thumb_contact_passed": actual_hand_passed,
            "actual_aero_native_thumb_force_n": float(full_native["finger_forces"][0]),
            "actual_aero_mjx_thumb_force_n": full_mjx_thumb_force,
            "actual_aero_contact_dimensions": actual_contact_dimensions,
            "actual_aero_sliding_friction": actual_contact_frictions,
            "actual_aero_mjx_thumb_sliding_friction": mjx_thumb_friction.tolist(),
            "native_contact_count": native_count,
            "native_first_normal_force": float(native_force[0]),
            "mjx_penetrating_contacts": int(active.sum()),
            "mjx_constraint_force_l1": force_sum}


def _quat_z_to(axis):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    if axis[2] < -.999999:
        return np.array([0., 1., 0., 0.])
    q = np.r_[1 + axis[2], -axis[1], axis[0], 0.]
    return q / np.linalg.norm(q)


def _plane(n):
    e1 = np.array([0., 0., 1.]) - n[2] * n
    if np.linalg.norm(e1) < .1:
        e1 = np.array([1., 0., 0.]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def candidate_initial_state(model, point, config, ids=None):
    """Return one deterministic reset pose, never an approved grasp.

    Placing the pen is permitted only at reset. This state can visualize a
    rejected search candidate but does not satisfy the preparation gate.
    """
    ids = ids or model_ids(model)
    d = _new_data(model)
    lo, hi = ids["ctrl_lower"].copy(), ids["ctrl_upper"].copy()
    # The abduction actuator's permissive controller range extends past its joint.
    lo[4] = max(lo[4], 0.0)
    hi[4] = min(hi[4], model.jnt_range[model.joint("right_thumb_cmc_abd").id, 1])
    d.ctrl[:] = lo + point[:7] * (hi - lo)
    qadr, vadr = ids["pen_qpos_adr"], ids["pen_dof_adr"]
    # The pen is only placed during reset; no support or applied forces are used.
    d.qpos[qadr:qadr+3] = [2, 2, 10]
    for _ in range(round(1 / config.sim_dt)):
        mujoco.mj_step(model, d)
    c, n = thumb_frame(model, d, ids)
    e1, e2 = _plane(n)
    radius = .015 + .015 * point[7]
    phase = 2 * np.pi * point[8]
    axial = .02 * (point[9] - .5)
    beta = math.radians(30) * (point[10] - .5)
    radial = np.cos(phase) * e1 + np.sin(phase) * e2
    axis = -np.sin(phase + beta) * e1 + np.cos(phase + beta) * e2
    d.qpos[qadr:qadr+3] = c + radius * radial + axial * n
    d.qpos[qadr+3:qadr+7] = _quat_z_to(axis)
    d.qvel[vadr:vadr+6] = 0
    mujoco.mj_forward(model, d)
    return d


def _candidate(model, ids, point, config, deadline):
    d = candidate_initial_state(model, point, config, ids)
    if not np.isfinite(d.qpos).all() or time.monotonic() >= deadline:
        return None, {"reason": "nonfinite_hand_or_budget"}
    c, _ = thumb_frame(model, d, ids)
    initial = native_contacts(model, d, ids)
    if initial["max_penetration"] > config.max_penetration or initial["invalid_support"]:
        return None, {"reason": "invalid_initial_contact",
                      "initial_penetration_m": initial["max_penetration"]}
    tail_hold = 0.0
    best_hold = 0.0
    speed_sum = 0.0
    detail = {}
    for i in range(round(2 / config.sim_dt)):
        if i % 100 == 0 and time.monotonic() >= deadline:
            return None, {"reason": "budget"}
        mujoco.mj_step(model, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            return None, {"reason": "nonfinite"}
        stable, detail = stable_sample(model, d, ids, config)
        if detail["invalid_support"]:
            return None, {"reason": "invalid_support"}
        if detail["max_penetration"] > config.max_penetration:
            return None, {"reason": "penetration",
                          "max_penetration_m": detail["max_penetration"]}
        if np.linalg.norm(d.site_xpos[ids["site_center"]] - c) > .2:
            return None, {"reason": "left_hand_region"}
        tail_hold = tail_hold + config.sim_dt if stable else 0.0
        best_hold = max(best_hold, tail_hold)
        speed_sum += detail["linear_speed"]
    c, n = thumb_frame(model, d, ids)
    record = {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "ctrl": d.ctrl.copy(),
              "c0": c, "n0": n}
    geometry = reference_geometry(model, d, ids, record, config)
    detail = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in detail.items()}
    reason = "unstable"
    if tail_hold + 1e-9 >= config.capture_seconds:
        reason = "passed" if geometry["passed"] else "initial_task_envelope"
    detail.update(reason=reason, geometry=geometry,
                  final_hold_seconds=tail_hold, best_hold_seconds=best_hold,
                  mean_linear_speed=speed_sum / round(2 / config.sim_dt))
    return record, detail


def search_grasp(model, config, max_candidates=2048, budget_seconds=900, seed=0):
    if max_candidates < 0 or max_candidates > 2048:
        raise ValueError("max_candidates must be in [0, 2048]")
    if not 0 < budget_seconds <= 900:
        raise ValueError("search budget must be in (0, 900] seconds")
    start = time.monotonic()
    deadline = start + budget_seconds
    sampler = qmc.Sobol(11, scramble=True, seed=seed)
    points = sampler.random_base2(int(np.ceil(np.log2(max(max_candidates, 1)))))
    ids = model_ids(model)
    counts, best, best_detail, best_score = {}, None, None, None
    examined = 0
    for point in points[:max_candidates]:
        if time.monotonic() >= deadline:
            break
        record, detail = _candidate(model, ids, point, config, deadline)
        examined += 1
        reason = detail["reason"]
        counts[reason] = counts.get(reason, 0) + 1
        if record is not None:
            score = (detail["final_hold_seconds"], detail["best_hold_seconds"],
                     -detail["mean_linear_speed"])
            if reason == "passed" or best_score is None or score > best_score:
                best, best_detail, best_score = record, detail, score
            if reason == "passed":
                break
        if examined % 64 == 0:
            print(f"prepare: {examined}/{max_candidates} candidates, {counts}", flush=True)
    return best, {
        "passed": best_detail is not None and best_detail["reason"] == "passed",
        "examined": examined, "max_candidates": max_candidates, "seed": seed,
        "elapsed_seconds": time.monotonic() - start,
        "budget_seconds": budget_seconds, "outcomes": counts, "best": best_detail,
    }


def check_reference_timesteps(reference, config):
    outcomes = []
    final_positions = []
    for dt in (config.sim_dt, config.sim_dt / 2):
        cfg = replace(config, sim_dt=dt)
        m = build_model(cfg); ids = model_ids(m); d = mujoco.MjData(m)
        d.qpos[:] = reference["qpos"]; d.qvel[:] = reference["qvel"]
        d.ctrl[:] = reference["ctrl"]; mujoco.mj_forward(m, d)
        tail = 0.0; invalid = False
        for _ in range(round(2 / dt)):
            mujoco.mj_step(m, d)
            stable, detail = stable_sample(m, d, ids, cfg)
            geometry_ok = reference_geometry(m, d, ids, reference, cfg)["passed"]
            tail = tail + dt if stable and geometry_ok else 0.0
            invalid |= detail["invalid_support"] or detail["max_penetration"] > config.max_penetration
        outcomes.append(bool(tail + 1e-9 >= config.capture_seconds and not invalid))
        final_positions.append(d.site_xpos[ids["site_center"]].copy())
    difference = float(np.linalg.norm(final_positions[0] - final_positions[1]))
    return {"passed": all(outcomes) and difference <= .005,
            "hold_passed": outcomes, "final_position_difference_m": difference,
            "position_tolerance_m": .005}


def run_prepare(output, config=None, max_candidates=2048, budget_seconds=900,
                seed=0, check_mjx=True):
    batch_started_unix = time.time()
    config = config or PenSpinConfig()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    # Preserve old successful outputs; a failed rerun must not leave a stale gate.
    if (output / "reference.npz").exists():
        raise FileExistsError("Use a new output directory; reference.npz already exists.")
    config.save(output / "config.json")
    model = build_model(config)
    report = {"schema_version": 1, "passed": False, "status": "failed",
              "batch_started_unix": batch_started_unix,
              "batch_deadline_unix": batch_started_unix + config.budget_hours * 3600,
              "config_digest": config.digest, "model_digest": model_fingerprint(config),
              "task_digest": task_fingerprint(config),
              "gates": {}, "search": None}
    report["gates"]["model"] = check_model(model)
    report["gates"]["freeflight_dt"] = check_freeflight(config)
    if check_mjx:
        try:
            report["gates"]["mjx_contact"] = check_mjx_contact(model)
        except Exception as e:
            report["gates"]["mjx_contact"] = {"passed": False, "error": f"{type(e).__name__}: {e}"}
    else:
        report["gates"]["mjx_contact"] = {"passed": False, "skipped": True,
                                         "reason": "diagnostic only: MJX gate not executed"}
    if all(g["passed"] for g in report["gates"].values()):
        first_point = qmc.Sobol(11, scramble=True, seed=seed).random_base2(0)[0]
        diagnostic = candidate_initial_state(model, first_point, config)
        c0, n0 = thumb_frame(model, diagnostic)
        np.savez(output / "diagnostic_reset.npz", qpos=diagnostic.qpos.copy(),
                 qvel=diagnostic.qvel.copy(), ctrl=diagnostic.ctrl.copy(), c0=c0, n0=n0,
                 approved=np.array(False), purpose=np.array("rejected-candidate visualization only"))
        report["diagnostic_reset"] = {"path": "diagnostic_reset.npz", "approved": False,
                                      "purpose": "first search reset; not an approved grasp"}
        ref, search = search_grasp(model, config, max_candidates, budget_seconds, seed)
        report["search"] = search
        if ref is not None:
            np.savez(output / "best_candidate.npz", **ref)
        if search["passed"]:
            report["gates"]["reference_geometry"] = check_reference_geometry(ref, config)
            report["gates"]["static_hold_dt"] = check_reference_timesteps(ref, config)
            if all(g["passed"] for g in report["gates"].values()):
                np.savez(output / "reference.npz", **ref,
                         config_digest=np.array(config.digest),
                         task_digest=np.array(report["task_digest"]),
                         model_digest=np.array(report["model_digest"]))
                report.update(passed=True, status="passed",
                              reference_sha256=hashlib.sha256((output / "reference.npz").read_bytes()).hexdigest())
    else:
        report["blocked_reason"] = "A numerical/model/MJX gate failed; grasp search was not started."
    if not report["passed"] and report["search"] is not None:
        report["blocked_reason"] = "No static-grasp reference passed every required gate."
    (output / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-candidates", type=int, default=2048)
    parser.add_argument("--budget-seconds", type=float, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-mjx-check", action="store_true",
                        help="Diagnostic only: always blocks an approved reference.")
    args = parser.parse_args()
    config = PenSpinConfig.load(args.config) if args.config else PenSpinConfig()
    report = run_prepare(args.output, config, args.max_candidates,
                         args.budget_seconds, args.seed, not args.skip_mjx_check)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
