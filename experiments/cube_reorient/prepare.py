"""Prepare a palm-up reference, preserving true free-body dynamics.

If no existing upright reference is given, regenerate the deterministic grasp
construction. Its rigid transform is only a reset proposal; the new gravity
orientation is then checked independently at 2 ms and 1 ms.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .config import CubeReorientConfig
from .model import build_model, model_fingerprint, model_ids, native_contacts, fixed_axis_basis
from experiments.cube_grasp.prepare import trial, search as upright_search
from experiments.cube_grasp.config import CubeGraspConfig


def prepare(output, source=None, seconds=300.):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'reference.npz').exists():
        raise FileExistsError('Choose a fresh output; approved references are immutable')
    if source is None:
        construction=output/'upright_construction'
        report=upright_search(construction,CubeGraspConfig(),seconds=seconds,seed=17)
        if not report['approved']:
            raise RuntimeError('Initial grasp construction did not pass its physical checks')
        source=construction/'reference.npz'
    source=Path(source).resolve()
    parent_report=json.loads((source.parent/'prepare_report.json').read_text())
    if not parent_report.get('approved'):
        raise ValueError('The source reference must pass preparation first')
    expected=parent_report.get('reference_sha256')
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if expected and source_hash!=expected:raise ValueError('Source reference checksum mismatch')
    config=CubeReorientConfig()
    model=build_model(config);ids=model_ids(model)
    with np.load(source,allow_pickle=False) as data:
        qpos=data['qpos'].copy();ctrl=data['ctrl'].copy()
    old_q=np.array([np.sqrt(.5),0,-np.sqrt(.5),0])
    inverse=old_q.copy();inverse[1:]*=-1
    delta=np.empty(4);mujoco.mju_mulQuat(delta,model.body('tetheria_mount').quat,inverse)
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,delta);rotation=rotation.reshape(3,3)
    center=model.body('tetheria_mount').pos.copy();q=ids['cube_qpos_adr']
    qpos[q:q+3]=center+rotation@(qpos[q:q+3]-center)
    result=np.empty(4);mujoco.mju_mulQuat(result,delta,qpos[q+3:q+7].copy());qpos[q+3:q+7]=result
    qvel=np.zeros(model.nv)
    checks={};ends=[]
    # Let the changed gravity orientation settle before defining an episode's
    # initial orientation. This passive transient is not rotation progress.
    settling, settled=trial(model,qpos,ctrl,seconds=3.,qvel=qvel,config=config)
    if not settling['success']:
        raise RuntimeError('The horizontal grasp did not survive passive settling')
    qpos=settled.qpos.copy()
    for dt in (config.sim_dt,config.sim_dt/2):
        physical=replace(config,sim_dt=dt)
        m=build_model(physical)
        row,end=trial(m,qpos,ctrl,seconds=3.,qvel=qvel,config=physical)
        checks[str(dt)]=row;ends.append(end.qpos[q:q+3].copy())
    agreement=float(np.linalg.norm(ends[0]-ends[1]))
    approved=all(row['success'] for row in checks.values()) and agreement<.005
    report=dict(approved=approved,task='horizontal_cube_reorient_xyz90',config=config.to_dict(),
                model_fingerprint=model_fingerprint(config),scope='Palm-up initial holding feasibility, not rotation success',
                construction=dict(source_reference=str(source),source_sha256=source_hash,
                                  method='rigid_transform_of_prepared_grasp_then_unassisted_free_release',
                                  quaternion_delta=delta.tolist(),velocities_reset_to_zero=True,
                                  passive_settling_seconds=3.,passive_settling=settling),
                mount_quaternion=model.body('tetheria_mount').quat.tolist(),
                palm_tilt_degrees=float(np.rad2deg(np.arccos(np.clip(fixed_axis_basis(model)[2,2],-1,1)))),
                axis_basis_world=fixed_axis_basis(model).tolist(),
                axes=dict(x='wrist to fingertips',y='index side to little-finger side',z='palm outward normal'),
                gravity=model.opt.gravity.tolist(),numerical_checks=checks,final_position_difference=agreement)
    if approved:
        np.savez(output/'reference.npz',qpos=qpos,qvel=qvel,ctrl=ctrl,grasp_ctrl=ctrl)
        report['reference_sha256']=hashlib.sha256((output/'reference.npz').read_bytes()).hexdigest()
    (output/'config.json').write_text(json.dumps(config.to_dict(),indent=2))
    (output/'prepare_report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--source',help='Optional approved upright reference; otherwise regenerate with seed 17')
    p.add_argument('--seconds',type=float,default=300.)
    args=p.parse_args();r=prepare(args.output,args.source,args.seconds)
    print(json.dumps(r,indent=2));return 0 if r['approved'] else 2


if __name__=='__main__':raise SystemExit(main())
