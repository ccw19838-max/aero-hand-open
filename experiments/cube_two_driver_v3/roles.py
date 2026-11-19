"""Measured contact wrenches. Never apply forces or change physical state."""
import numpy as np
import mujoco

SUPPORT=(1,2,3)
DRIVERS=(0,4)


def support_state(forces):
    f=np.asarray(forces,dtype=float)[list(SUPPORT)]
    return bool(np.all(f>=.1) and np.all(f<=12.))


def wrenches(env):
    model,data,ids=env.model,env.data,env.ids
    center=data.xipos[ids['cube_body']];direction=env.monitor.direction
    forces=np.zeros((5,3));moments=np.zeros((5,3));speeds=np.zeros(5);weights=np.zeros(5)
    wrench=np.zeros(6);jac=np.zeros((3,model.nv));center_jac=np.zeros_like(jac)
    mujoco.mj_jac(model,data,center_jac,None,center,ids['cube_body']);center_velocity=center_jac@data.qvel
    for index,contact in enumerate(data.contact):
        g1,g2=int(contact.geom1),int(contact.geom2)
        if ids['cube_geom'] not in (g1,g2):continue
        other=g1 if g2==ids['cube_geom'] else g2;finger=int(ids['geom_finger'][other])
        if finger<0:continue
        mujoco.mj_contactForce(model,data,index,wrench)
        sign=1. if g2==ids['cube_geom'] else -1.;frame=contact.frame.reshape(3,3)
        force=sign*(frame.T@wrench[:3]);arm=contact.pos-center
        forces[finger]+=force;moments[finger]+=np.cross(arm,force)+sign*(frame.T@wrench[3:])
        tangent=np.cross(direction,arm);length=np.linalg.norm(tangent)
        if wrench[0]>0. and length>1e-8:
            mujoco.mj_jac(model,data,jac,None,contact.pos,int(model.geom_bodyid[other]))
            speeds[finger]+=float(wrench[0])*float((jac@data.qvel-center_velocity)@(tangent/length));weights[finger]+=float(wrench[0])
    return dict(axis_moments_Nm=moments@direction,tangential_speeds_m_s=speeds/np.maximum(weights,1e-12),forces_world_N=forces)


def driving_quality(moment,speed):
    # Static squeeze or reverse moment cannot earn drive credit.
    return float(np.clip(moment/.005,0.,1.)*np.clip(speed/.005,0.,1.))
