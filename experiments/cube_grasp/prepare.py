"""Construct grasp resets, then validate them without fixtures or external forces.

Fixture-assisted pose construction is never a learned-policy evaluation. Approved
references describe pregrasp holding, not acquisition from an open hand.
"""
import argparse
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from .model import (build_model, model_ids, model_fingerprint, native_contacts,
                    native_cube_velocity, scene_and_assets, HOME_QPOS)
from .config import CubeGraspConfig


class SearchDeadline(Exception):
    """The preparation wall-clock budget expired."""


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise SearchDeadline


def _write_report(output, report):
    path = Path(output) / 'prepare_report.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def _state(model, qpos, ctrl, qvel=None):
    """Fresh free-body state; no auxiliary constraints, applied forces or warmstart."""
    qpos = np.asarray(qpos, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)
    qvel = np.zeros(model.nv) if qvel is None else np.asarray(qvel, dtype=float)
    if qpos.shape != (model.nq,) or qvel.shape != (model.nv,) or ctrl.shape != (model.nu,):
        raise ValueError('Candidate state dimensions do not match the free-body model')
    if not all(np.isfinite(x).all() for x in (qpos, qvel, ctrl)):
        raise ValueError('Candidate state must be finite')
    if np.any(ctrl < model.actuator_ctrlrange[:, 0]) or np.any(ctrl > model.actuator_ctrlrange[:, 1]):
        raise ValueError('Candidate actuator targets exceed their limits')
    ids = model_ids(model)
    adr = ids['cube_qpos_adr']
    if not math.isclose(float(np.linalg.norm(qpos[adr+3:adr+7])), 1., abs_tol=1e-5):
        raise ValueError('Candidate cube quaternion must have unit length')
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    return data, ids


def trial(model, qpos, ctrl, seconds=3., qvel=None, config=None, deadline=None):
    """Full free-body rollout, checking contacts at every physical substep."""
    config = config or CubeGraspConfig()
    data, ids = _state(model, qpos, ctrl, qvel)
    adr = ids['cube_qpos_adr']
    start = data.qpos[adr:adr+3].copy()
    contact = native_contacts(model, data, ids)
    initial_penetration = float(contact['max_penetration'])
    stable = best = 0.
    peak = initial_penetration
    support = bool(contact['forbidden_support'])
    finite = True
    max_displacement = 0.
    failure = None
    for _ in range(round(seconds / model.opt.timestep)):
        _check_deadline(deadline)
        if support:
            failure = 'forbidden_support'
            break
        mujoco.mj_step(model, data)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        if not finite:
            failure = 'nonfinite'
            break
        contact = native_contacts(model, data, ids)
        velocity, omega = native_cube_velocity(model, data, ids)
        peak = max(peak, float(contact['max_penetration']))
        support |= bool(contact['forbidden_support'])
        displacement = float(np.linalg.norm(data.qpos[adr:adr+3] - start))
        max_displacement = max(max_displacement, displacement)
        fingers = contact['finger_forces'] >= config.contact_force
        good = (fingers[0] and np.count_nonzero(fingers) >= config.min_fingers
                and np.linalg.norm(velocity) < config.grip_velocity
                and np.linalg.norm(omega) < config.grip_omega
                and contact['max_penetration'] <= config.max_penetration
                and not support and displacement <= config.max_displacement)
        stable = stable + model.opt.timestep if good else 0.
        best = max(best, stable)
        if support:
            failure = 'forbidden_support'
        elif peak > config.max_penetration:
            failure = 'penetration'
        elif displacement > config.max_displacement:
            failure = 'workspace'
        if failure:
            break
    complete = data.time >= seconds - model.opt.timestep * .1
    success = (complete and finite and not failure and not support
               and peak <= config.max_penetration
               and stable >= config.hold_seconds - model.opt.timestep * .1)
    row = dict(best_hold_seconds=float(best), final_hold_seconds=float(stable),
               peak_penetration=float(peak), initial_penetration=initial_penetration,
               forbidden_support=support, finite=finite, failure_reason=failure,
               displacement=float(np.linalg.norm(data.qpos[adr:adr+3] - start)),
               max_displacement=max_displacement, finger_forces=contact['finger_forces'].tolist(),
               seconds=float(data.time), requested_seconds=seconds, completed=bool(complete),
               success=bool(success))
    return row, data


def approve(output, config, qpos, ctrl, attempts=None, calls=0, *, qvel=None,
            construction=None, deadline=None):
    """Validate a candidate at both timesteps; save only genuinely approved state.

    The original positional API is retained for diagnostic scripts. A candidate
    above the configured physical penetration limit can never be approved.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tests = {}
    ends = []
    reason = None
    model = build_model(config)
    state, ids = _state(model, qpos, ctrl, qvel)
    qvel = state.qvel.copy()
    initial = native_contacts(model, state, ids)
    reset_valid = (initial['max_penetration'] <= config.max_penetration
                   and not initial['forbidden_support'])
    if not reset_valid:
        reason = 'Initial state violates configured reset validity thresholds'
    else:
        try:
            for dt in (config.sim_dt, config.sim_dt / 2):
                _check_deadline(deadline)
                current = build_model(replace(config, sim_dt=dt))
                row, end = trial(current, qpos, ctrl, seconds=3., qvel=qvel,
                                 config=config, deadline=deadline)
                tests[str(dt)] = row
                adr = model_ids(current)['cube_qpos_adr']
                ends.append(end.qpos[adr:adr+3].copy())
        except SearchDeadline:
            reason = 'Preparation time budget exhausted during independent validation'
    agreement = float(np.linalg.norm(ends[0] - ends[1])) if len(ends) == 2 else None
    approved = bool(reset_valid and len(tests) == 2
                    and all(row['success'] for row in tests.values()) and agreement < .005)
    if not approved and reason is None:
        reason = 'Free-body holding or timestep agreement check failed'
    report = dict(approved=approved, config=config.to_dict(),
                  model_fingerprint=model_fingerprint(config), search_calls=calls,
                  scope='Prepared pregrasp holding feasibility; not active acquisition or learned-policy success',
                  construction=construction or {'method': 'externally_provided_candidate',
                                               'details': 'Construction provenance was not supplied'},
                  validation='Fresh free-body MjData; no fixture, applied force, or auxiliary warmstart',
                  initial_penetration=float(initial['max_penetration']),
                  reset_penetration_limit=config.max_penetration,
                  contact_tolerance_note='Native soft-contact penetration uses the same configured limit at reset and during the rollout',
                  numerical_checks=tests, final_position_difference=agreement,
                  final_position_difference_limit=.005, reason=reason, attempts=attempts or [])
    if approved:
        temporary = output / 'reference.tmp.npz'
        np.savez(temporary, qpos=np.asarray(qpos), qvel=qvel,
                 ctrl=np.asarray(ctrl), grasp_ctrl=np.asarray(ctrl))
        reference = output / 'reference.npz'
        temporary.replace(reference)
        report['reference_sha256'] = hashlib.sha256(reference.read_bytes()).hexdigest()
    _write_report(output, report)
    return report


class FixtureConstructor:
    """A temporary static cube supplies contact constraints only during setup."""
    def __init__(self, config):
        xml, assets = scene_and_assets(config)
        root = ET.fromstring(xml)
        cube = root.find('.//body[@name="cube"]')
        cube.remove(cube.find('freejoint'))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets)
        self.data = mujoco.MjData(self.model)
        self.cube_body = self.model.body('cube').id

    def pose(self, position, quaternion, ctrl, hand_qpos=None, seconds=1.5, deadline=None):
        _check_deadline(deadline)
        m, d = self.model, self.data
        m.body_pos[self.cube_body] = position
        m.body_quat[self.cube_body] = quaternion
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = HOME_QPOS if hand_qpos is None else hand_qpos
        d.ctrl[:] = ctrl
        # Small chunks bound deadline overshoot without a Python call per step.
        steps = round(seconds / m.opt.timestep)
        while steps:
            _check_deadline(deadline)
            chunk = min(steps, 50)
            mujoco.mj_step(m, d, nstep=chunk)
            steps -= chunk
        return np.r_[d.qpos.copy(), position, quaternion]


def _failed_report(output, config, calls, attempts, construction, started, seconds):
    report = dict(approved=False, config=config.to_dict(), model_fingerprint=model_fingerprint(config),
                  search_calls=calls, construction=construction,
                  scope='Initial-state construction only; no trained policy was evaluated',
                  reason='No candidate passed all free-body checks within the preparation budget',
                  elapsed_seconds=time.monotonic()-started, budget_seconds=seconds, attempts=attempts)
    _write_report(output, report)
    return report


def fixture_search(output, config, seconds=300., seed=17):
    """Deterministic fixture poses plus bounded preload relaxation and validation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + seconds
    fixture = FixtureConstructor(config)
    free = build_model(config)
    ids = model_ids(free)
    if fixture.model.nq != ids['cube_qpos_adr'] or free.nq != fixture.model.nq + 7:
        raise ValueError('Unexpected joint ordering in fixture construction')
    rng = np.random.default_rng(seed)
    calls = 0
    attempts = []
    best = -float('inf')
    preferred = min(.0004, config.reset_max_penetration)
    construction = dict(method='temporary_static_cube_then_free_release', seed=seed,
                        budget_seconds=seconds, initial_fixture_seconds=1.5,
                        refinement_fixture_seconds=1.,
                        details='Cube freejoint removed only in the setup model; all approval rollouts use the original free-body model. The constructed pose may already be grasping.')

    def consider(qpos, ctrl, phase, index):
        nonlocal calls, best
        _check_deadline(deadline)
        calls += 1
        state, _ = _state(free, qpos, ctrl)
        initial = native_contacts(free, state, ids)
        if initial['forbidden_support'] or initial['max_penetration'] > config.max_penetration:
            return None, False
        row, _ = trial(free, qpos, ctrl, config=config, deadline=deadline)
        row.update(phase=phase, candidate_index=index, ctrl=np.asarray(ctrl).tolist())
        attempts.append(row)
        score = row['best_hold_seconds'] + min(row['seconds'], 3.)*.05
        if score > best:
            best = score
            np.savez(output/'candidate.npz', qpos=qpos, qvel=np.zeros(free.nv),
                     ctrl=ctrl, grasp_ctrl=ctrl)
            print(json.dumps(dict(search_calls=calls, best=row)), flush=True)
        if row['success'] and initial['max_penetration'] <= config.max_penetration:
            result = approve(output, config, qpos, ctrl, attempts, calls,
                             construction=construction, deadline=deadline)
            if result['approved']:
                result.update(elapsed_seconds=time.monotonic()-started, budget_seconds=seconds)
                _write_report(output, result)
                return result, True
        return None, bool(row['success'])

    try:
        for index in range(400):
            _check_deadline(deadline)
            if index < 30:
                position = np.array([[-.03, -.02, -.01][index % 3], -.005,
                                     [.265, .275, .285][index//3 % 3]])
                ctrl = np.r_[np.full(4, [.075, .08, .085][index//9 % 3]), .8, .033, .097]
            else:
                position = rng.uniform([-.04, -.025, .255], [-.005, .015, .31])
                ctrl = np.r_[np.clip(np.full(4, rng.uniform(.069, .09)) + rng.normal(0, .0015, 4), .065, .095),
                             rng.uniform(.2, 1.45), rng.uniform(.0275, .0375), rng.uniform(.087, .108)]
            ctrl = np.clip(ctrl, ids['ctrl_lower'], ids['ctrl_upper'])
            quaternion = np.array([1., 0., 0., 0.])
            qpos = fixture.pose(position, quaternion, ctrl, deadline=deadline)
            result, physically_stable = consider(qpos, ctrl, 'fixture', index)
            if result is not None:
                return result
            if not physically_stable:
                continue
            # A stable but overly preloaded contact pose is relaxed by rebuilding
            # it under the same fixture. No penetration threshold is loosened.
            for refinement in range(180):
                _check_deadline(deadline)
                target = ctrl.copy()
                if refinement < 30:
                    target[[0, 1, 2, 3, 6]] += .0005*(refinement % 6)
                    target[4] -= .025*(refinement//6)
                else:
                    target += rng.normal(0, [.002, .002, .002, .002, .06, .0008, .002])
                target = np.clip(target, ids['ctrl_lower'], ids['ctrl_upper'])
                relaxed = fixture.pose(position, quaternion, target,
                                       hand_qpos=qpos[:ids['cube_qpos_adr']],
                                       seconds=1., deadline=deadline)
                state, _ = _state(free, relaxed, target)
                if native_contacts(free, state, ids)['max_penetration'] > preferred:
                    continue
                result, _ = consider(relaxed, target, f'refine_{index}', refinement)
                if result is not None:
                    return result
    except SearchDeadline:
        pass
    return _failed_report(output, config, calls, attempts, construction, started, seconds)


def geometry_search(output, config, seconds=300., seed=17):
    """Optional fixture-free geometry search; retained as an independent method."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + seconds
    model = build_model(config)
    ids = model_ids(model)
    data = mujoco.MjData(model)
    tips = [model.geom(n).id for n in ('th_tip', 'if_tip', 'mf_tip', 'rf_tip', 'pf_tip')]
    colliders = [g for g in range(model.ngeom)
                 if (model.geom_contype[g] or model.geom_conaffinity[g]) and g != ids['cube_geom']]
    archive = []
    attempts = []
    calls = 0
    construction = dict(method='fixture_free_geometry', seed=seed, budget_seconds=seconds)

    def pose(x):
        _check_deadline(deadline)
        mujoco.mj_resetData(model, data)
        adr = ids['cube_qpos_adr']
        data.qpos[:adr] = HOME_QPOS
        data.qpos[adr:] = [1, 0, 10, 1, 0, 0, 0]
        data.ctrl[:] = np.clip([x[0], x[0], x[0]+.002, x[0]+.004, x[1], x[2], x[3]],
                               ids['ctrl_lower'], ids['ctrl_upper'])
        mujoco.mj_step(model, data, nstep=250)
        data.qpos[adr:] = [*x[4:7], np.cos(x[7]/2), 0, 0, np.sin(x[7]/2)]
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)

    def objective(x):
        nonlocal calls
        calls += 1
        pose(x)
        distances = np.array([mujoco.mj_geomDistance(model, data, ids['cube_geom'], g, .15, None) for g in tips])
        gaps = np.array([mujoco.mj_geomDistance(model, data, ids['cube_geom'], g, .15, None) for g in colliders])
        loss = float(np.sum(((distances[:3]-.00015)/.01)**2)
                     + 40*np.sum((np.minimum(gaps+.0002, 0)/.01)**2))
        if loss < .15:
            archive.append((loss, data.qpos.copy(), data.ctrl.copy()))
            archive.sort(key=lambda item:item[0])
            del archive[40:]
        return loss

    bounds = [(.069, .092), (.1, 1.6), (.027, .038), (.083, .11),
              (-.08, -.025), (-.025, .01), (.27, .325), (-.35, .35)]
    try:
        for restart in range(8):
            result = differential_evolution(objective, bounds, popsize=5, maxiter=18,
                                            seed=seed+restart, polish=True, tol=.01)
            pose(result.x)
            archive.append((result.fun, data.qpos.copy(), data.ctrl.copy()))
            candidates = sorted(archive, key=lambda item:item[0])[:12]
            archive = []
            for loss, qpos, ctrl in candidates:
                for preload in (0., .0005, .001, .002):
                    _check_deadline(deadline)
                    target = ctrl.copy()
                    target[:4] -= preload
                    target[6] -= preload*.5
                    target = np.clip(target, ids['ctrl_lower'], ids['ctrl_upper'])
                    row, _ = trial(model, qpos, target, config=config, deadline=deadline)
                    row.update(geometric_loss=float(loss), preload=preload)
                    attempts.append(row)
                    if row['success']:
                        report = approve(output, config, qpos, target, attempts, calls,
                                         construction=construction, deadline=deadline)
                        if report['approved']:
                            return report
    except SearchDeadline:
        pass
    return _failed_report(output, config, calls, attempts, construction, started, seconds)


def search(output, config, seconds=300., seed=17, method='fixture'):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Preparation budget must be positive and finite')
    if method == 'fixture':
        return fixture_search(output, config, seconds, seed)
    if method == 'geometry':
        return geometry_search(output, config, seconds, seed)
    raise ValueError(f'Unknown preparation method: {method}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--method', choices=('fixture', 'geometry'), default='fixture')
    parser.add_argument('--candidate', type=Path, help='Validate an existing NPZ without searching or using a fixture')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive and finite')
    config = CubeGraspConfig()
    if args.candidate:
        with np.load(args.candidate, allow_pickle=False) as candidate:
            ctrl = candidate['grasp_ctrl'] if 'grasp_ctrl' in candidate else candidate['ctrl']
            qvel = candidate['qvel'] if 'qvel' in candidate else None
            result = approve(args.output, config, candidate['qpos'], ctrl, qvel=qvel,
                             construction=dict(method='candidate_validation', source=str(args.candidate.resolve()),
                                               source_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                                               details='No assistance during validation; candidate construction history is external'),
                             deadline=time.monotonic()+args.seconds)
    else:
        result = search(args.output, config, args.seconds, args.seed, args.method)
    print(json.dumps({key:value for key,value in result.items() if key != 'attempts'}, indent=2))
    return 0 if result['approved'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
