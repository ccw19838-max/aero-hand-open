"""Unmodified Aero tendons with OpenAI robogym's locked Rubik cube.

Upstream cube XML, texture, MIT license and pinned source hashes are in assets/.
Layer turns are deliberately absent: the object is one free rigid body.
"""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HAND_XML = ROOT / 'sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/right_hand.xml'
HAND_ASSETS = ROOT / 'sim_rl/simulation/mujoco/assets'
CUBE_ASSETS = Path(__file__).parent / 'assets/robogym'
HOME_QPOS = np.array([1.57335,.347305,.352258,1.57303,.400789,.406,1.57335,.365219,.370336,1.57336,.515229,.520112,.746438,.260182,-.00155382,1.0748])
HOME_CTRL = np.array([.09,.09,.09,.09,.75,.035,.1])


def scene_and_assets(config=None):
    def val(name, default):
        return getattr(config, name, default)
    root = ET.fromstring(HAND_XML.read_text())
    root.set('model', 'aero_upright_rubik_grasp')
    option = root.find('option')
    option.set('timestep', str(val('sim_dt', .002)))
    option.set('gravity', '0 0 -9.81')
    option.set('iterations', str(val('solver_iterations', 20)))
    mount = root.find('.//body[@name="tetheria_mount"]')
    mount.set('pos', '0 0 .15')
    mount.set('quat', '.7071067811865476 0 -.7071067811865476 0')
    source = ET.fromstring((CUBE_ASSETS / 'rubik_locked.xml').read_text())
    asset = root.find('asset')
    for item in source.find('asset'):
        if item.tag == 'texture':
            item.set('file', 'rubik.png')
        asset.append(item)
    world = root.find('worldbody')
    ET.SubElement(world,'geom',name='floor',type='plane',size='1 1 .01',pos='0 0 0',rgba='.18 .20 .24 1',contype='2',conaffinity='3')
    ET.SubElement(world,'camera',name='overview',pos='-.43 -.48 .42',xyaxes='.745 -.667 0 .198 .221 .955')
    cube = ET.SubElement(world,'body',name='cube',pos='0 0 .6')
    ET.SubElement(cube,'freejoint',name='cube_freejoint')
    half = val('cube_size', .057) / 2
    geom = source.find('.//geom[@name="cube:middle"]')
    if geom is None:
        geom = next(g for g in source.findall('.//geom') if 'annotation' not in g.get('name',''))
    geom.set('name','cube_collision')
    geom.set('size',f'{half} {half} {half}')
    geom.set('mass',str(val('cube_mass', .077966253)))
    geom.set('condim','3')
    geom.set('friction',f"{val('cube_friction', .6)} .001 .0001")
    geom.set('contype','2'); geom.set('conaffinity','3')
    geom.set('priority','1')
    geom.set('solref','.02 1.5'); geom.set('solimp','.9 .99 .001')
    cube.append(geom)
    ET.SubElement(cube,'site',name='cube_center',size='.001',rgba='0 0 0 0')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth=str(max(1120, val('render_width', 960))),offheight=str(max(840, val('render_height', 720))))
    ET.SubElement(visual,'headlight',diffuse='.7 .7 .7',ambient='.4 .4 .4')
    assets = {p.name:p.read_bytes() for p in HAND_ASSETS.rglob('*') if p.is_file()}
    assets['rubik.png']=(CUBE_ASSETS/'rubik.png').read_bytes()
    return ET.tostring(root, encoding='unicode'),assets


def build_model(config=None):
    xml, assets = scene_and_assets(config)
    return mujoco.MjModel.from_xml_string(xml, assets)


def model_fingerprint(config=None):
    xml, assets = scene_and_assets(config)
    h=hashlib.sha256(xml.encode())
    for name, value in sorted(assets.items()):
        h.update(name.encode()); h.update(value)
    return h.hexdigest()


def model_ids(model):
    fingers=np.full(model.ngeom,-1,dtype=np.int32)
    forbidden=np.zeros(model.ngeom,dtype=bool)
    for g in range(model.ngeom):
        body=model.body(int(model.geom_bodyid[g])).name or ''
        for index,prefix in enumerate(('right_thumb','right_index','right_middle','right_ring','right_pinky')):
            if body.startswith(prefix): fingers[g]=index
        forbidden[g]=body=='tetheria_mount' or model.geom(g).name=='floor'
    joint=model.joint('cube_freejoint').id
    qa=int(model.jnt_qposadr[joint]); va=int(model.jnt_dofadr[joint])
    return dict(cube_body=model.body('cube').id,cube_geom=model.geom('cube_collision').id,
                cube_qpos_adr=qa,cube_dof_adr=va,hand_qpos_ids=np.arange(qa),hand_dof_ids=np.arange(va),
                ctrl_lower=model.actuator_ctrlrange[:,0].copy(),ctrl_upper=model.actuator_ctrlrange[:,1].copy(),
                geom_finger=fingers,geom_forbidden=forbidden)


def native_contacts(model, data, ids=None):
    ids=ids or model_ids(model)
    forces=np.zeros(5); forbidden=False; depth=0.; force=np.zeros(6)
    for i,c in enumerate(data.contact):
        g1,g2=int(c.geom1),int(c.geom2)
        other=g2 if g1==ids['cube_geom'] else g1 if g2==ids['cube_geom'] else None
        if other is None: continue
        mujoco.mj_contactForce(model,data,i,force)
        normal=max(0.,float(force[0]))
        finger=ids['geom_finger'][other]
        if finger>=0: forces[finger]+=normal
        forbidden|=bool(ids['geom_forbidden'][other] and normal>1e-6)
        depth=max(depth,-float(c.dist))
    return dict(finger_forces=forces,forbidden_support=forbidden,max_penetration=depth)


def native_cube_velocity(model, data, ids=None):
    ids=ids or model_ids(model); v=np.zeros(6)
    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,ids['cube_body'],v,0)
    return v[3:].copy(), v[:3].copy()
