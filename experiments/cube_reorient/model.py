"""Palm-up Aero and the same pinned rigid OpenAI Rubik cube.

The old upright task is unchanged. The mount uses its original rotation, retaining
the internal 20-degree palm tilt as requested. Hand XYZ stay fixed while
the object moves; none of the face colours are treated as interchangeable.
"""
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from experiments.cube_grasp.model import (
    scene_and_assets as upright_scene, model_ids, native_contacts,
    native_cube_velocity, HOME_QPOS, HOME_CTRL,
)


def scene_and_assets(config=None):
    xml, assets = upright_scene(config)
    root = ET.fromstring(xml)
    root.set('model', 'aero_horizontal_rubik_xyz90')
    root.find('.//body[@name="tetheria_mount"]').set('quat', '1 0 0 0')
    # Gravity is unchanged: this is a physical mounting change.
    world = root.find('worldbody')
    base = (.04, -.12, .15)
    basis=_palm_basis(root)
    for i,(axis,color) in enumerate([('x','1 .1 .1 1'),('y','.1 .9 .1 1'),('z','.2 .4 1 1')]):
        end=tuple(np.asarray(base)+.06*basis[:,i])
        ET.SubElement(world, 'geom', name=f'fixed_{axis}_axis', type='capsule',
                      fromto=' '.join(map(str, base + end)), size='.001',
                      rgba=color, contype='0', conaffinity='0', mass='0', group='2')
    return ET.tostring(root, encoding='unicode'), assets


def _palm_basis(root):
    q=np.fromstring(root.find('.//body[@name="palm"]').get('quat'),sep=' ')
    q/=np.linalg.norm(q);rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,q)
    matrix=rotation.reshape(3,3)
    return np.column_stack((matrix[:,2],-matrix[:,0],-matrix[:,1]))


def fixed_axis_basis(model):
    """World vectors of the immutable palm axes, not live finger/object axes."""
    root=ET.fromstring(upright_scene()[0])
    local=_palm_basis(root)
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,model.body('tetheria_mount').quat)
    return rotation.reshape(3,3)@local


def build_model(config=None):
    xml, assets = scene_and_assets(config)
    return mujoco.MjModel.from_xml_string(xml, assets)


def model_fingerprint(config=None):
    xml, assets = scene_and_assets(config)
    digest=hashlib.sha256(xml.encode())
    for name, content in sorted(assets.items()):
        digest.update(name.encode()); digest.update(content)
    return digest.hexdigest()
