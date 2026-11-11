"""Prepare collision-safe release states for active acquisition, not stable grips."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.stats import qmc
from .config import PenSpinConfig
from .model import build_model,model_ids,model_fingerprint,task_fingerprint,native_contacts,thumb_frame
from .prepare import candidate_initial_state,check_model,check_freeflight,check_mjx_contact
from .policy import digest_file


def prepare_grasp(output,candidates=2048):
    output=Path(output)
    if (output/'reference.npz').exists():
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    cfg=replace(PenSpinConfig(),schema_version=2,task_mode='grasp',budget_hours=2.,
                reset_position_noise=.00001,reset_angle_noise_degrees=.01)
    cfg.save(output/'config.json')
    start=time.time()
    m=build_model(cfg);ids=model_ids(m)
    gates={'model':check_model(m),'freeflight_dt':check_freeflight(cfg),'mjx_contact':check_mjx_contact(m)}
    best=None;best_score=float('inf');safe=0
    if all(g['passed'] for g in gates.values()):
        points=qmc.Sobol(11,scramble=True,seed=0).random_base2(int(np.ceil(np.log2(candidates))))[:candidates]
        for index,point in enumerate(points):
            if time.time()-start>900:break
            d=candidate_initial_state(m,point,cfg,ids)
            # The acquisition reset follows the fingertips, rather than forcing
            # the pen to orbit the thumb before it has even been grasped.
            from .prepare import _quat_z_to,_plane
            tips=np.array([d.site_xpos[m.site(n).id].copy()
                           for n in ('th_tip','if_tip','mf_tip','rf_tip','pf_tip')])
            pairs=((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
            pair=pairs[min(5,int(point[10]*6))]
            selected=tips[[0,*pair]]
            center=selected.mean(axis=0)
            _,_,vectors=np.linalg.svd(selected-center)
            axis=vectors[0]
            u,v=_plane(axis)
            offset=(.005+.007*point[7])*(np.cos(2*np.pi*point[8])*u+np.sin(2*np.pi*point[8])*v)
            center+=offset+.025*(point[9]-.5)*axis
            adr=ids['pen_qpos_adr']
            d.qpos[adr:adr+3]=center;d.qpos[adr+3:adr+7]=_quat_z_to(axis)
            mujoco.mj_forward(m,d)
            contact=native_contacts(m,d,ids)
            if contact['invalid_support'] or contact['max_penetration']>.0002:continue
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():continue
            safe+=1
            a,b=d.site_xpos[ids['site_a']],d.site_xpos[ids['site_b']]
            fractions=np.clip(((tips-b)*(a-b)).sum(axis=-1)/np.dot(a-b,a-b),0,1)
            distances=np.maximum(0,np.linalg.norm(tips-(b+fractions[:,None]*(a-b)),axis=-1)-.005)
            score=2*distances[0]+np.sort(distances[1:])[:2].sum()
            # Allow at least two control decisions before rejecting a release.
            # This is a brief viability test, not the old 2s static-grip gate.
            reset=dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy())
            clean=mujoco.MjData(m)
            clean.qpos[:]=reset['qpos'];clean.qvel[:]=reset['qvel'];clean.ctrl[:]=reset['ctrl']
            mujoco.mj_forward(m,clean)
            viable=True
            for _ in range(round(.1/cfg.sim_dt)):
                mujoco.mj_step(m,clean)
                c=native_contacts(m,clean,ids)
                if c['invalid_support'] or c['max_penetration']>cfg.max_penetration:
                    viable=False;break
            if not viable:continue
            if score<best_score:
                d.qpos[:]=reset['qpos'];d.qvel[:]=reset['qvel'];d.ctrl[:]=reset['ctrl']
                mujoco.mj_forward(m,d)
                c0,n0=thumb_frame(m,d,ids)
                # Holding may move the thumb: use pen center as the grasp workspace origin.
                best=dict(**reset,c0=d.site_xpos[ids['site_center']].copy(),n0=n0)
                best_score=score;best_distances=distances;best_index=index
            if index%128==0:print(f'grasp resets: examined={index+1},safe={safe},best_distance_score={best_score:.6f}',flush=True)
    gates['reset_geometry']={'passed':best is not None,'safe_resets':safe,'penetration_limit_m':.0002,
                             'unsupported_viability_seconds':.1}
    passed=all(g['passed'] for g in gates.values())
    report=dict(passed=passed,status='grasp_reset_ready' if passed else 'failed',
                scope='grasp_acquisition_only',stable_grasp=False,gates=gates,
                config_digest=cfg.digest,model_digest=model_fingerprint(cfg),task_digest=task_fingerprint(cfg),
                batch_started_unix=start,batch_deadline_unix=start+cfg.budget_hours*3600,
                candidates_requested=candidates,elapsed_seconds=time.time()-start)
    if passed:
        np.savez(output/'reference.npz',**best,config_digest=cfg.digest,
                 model_digest=report['model_digest'],task_digest=report['task_digest'])
        report.update(reference_sha256=digest_file(output/'reference.npz'),
                      selected_candidate=best_index,finger_surface_distances_m=best_distances.tolist())
    (output/'prepare_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidates',type=int,default=2048)
    args=parser.parse_args()
    if not 1<=args.candidates<=2048:parser.error('candidates must be in [1,2048]')
    report=prepare_grasp(args.output,args.candidates)
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
