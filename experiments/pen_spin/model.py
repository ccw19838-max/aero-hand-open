"""Upright Aero model and a parameterised, explicitly weighted pen.

The hand assets retain their upstream notices. The pen is newly specified
primitive geometry, inspired by segmented pen designs, not a copied mesh.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HAND_XML = REPO_ROOT / "sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/right_hand.xml"
ASSET_DIR = REPO_ROOT / "sim_rl/simulation/mujoco/assets"
PEN_LENGTH = 0.130
PEN_RADIUS = 0.005
PEN_MASS = 0.016
PEN_COM = np.array([0.0, 0.0, 0.006875])
PEN_INERTIA = np.array([3.874375e-5, 3.874375e-5, 2.0e-7])
HOME_HAND_QPOS = np.array([
    1.57335, .347305, .352258, 1.57303, .400789, .406,
    1.57335, .365219, .370336, 1.57336, .515229, .520112,
    .746438, .260182, -.00155382, 1.0748,
])
HOME_CTRL = np.array([.09, .09, .09, .09, .75, .035, .1])
TASK_CONFIG_FIELDS = (
    "schema_version", "sim_dt", "ctrl_dt", "episode_seconds", "contact_force",
    "grip_velocity", "grip_omega", "capture_seconds", "hold_seconds",
    "initial_hold_seconds", "free_contact_seconds", "violation_seconds",
    "max_plane_degrees", "soft_plane_degrees", "max_pivot_drift", "max_thumb_axis_degrees",
    "max_center_distance", "max_axial_distance", "min_endpoint_radius",
    "contact_sector_seconds", "angle_tolerance_degrees",
    "capture_direction_degrees", "max_penetration", "action_delta_fraction",
    "reset_position_noise", "reset_angle_noise_degrees",
)
HAND_JOINTS = [
    "right_index_mcp_flex", "right_index_pip", "right_index_dip",
    "right_middle_mcp_flex", "right_middle_pip", "right_middle_dip",
    "right_ring_mcp_flex", "right_ring_pip", "right_ring_dip",
    "right_pinky_mcp_flex", "right_pinky_pip", "right_pinky_dip",
    "right_thumb_cmc_abd", "right_thumb_cmc_flex", "right_thumb_mcp",
    "right_thumb_ip",
]


def scene_and_assets(config=None) -> tuple[str, dict[str, bytes]]:
    """Return the complete scene and VFS resources, leaving submodules untouched."""
    if not HAND_XML.is_file() or not ASSET_DIR.is_dir():
        raise FileNotFoundError("Initialize the pinned sim_rl submodules first.")
    root = ET.fromstring(HAND_XML.read_text())
    root.set("model", "aero_right_pen_spin")
    option = root.find("option")
    option.set("timestep", str(getattr(config, "sim_dt", .002)))
    option.set("gravity", "0 0 -9.81")
    # Contact limits must not silently reproduce the small cube task's capacity.
    for numeric in root.findall("custom/numeric"):
        if numeric.get("name") == "max_contact_points":
            numeric.set("data", "128")
        if numeric.get("name") == "max_geom_pairs":
            numeric.set("data", "128")
    mount = root.find('.//body[@name="tetheria_mount"]')
    mount.set("pos", "0 0 0.15")
    mount.set("quat", "0.7071067811865476 0 -0.7071067811865476 0")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane",
                  pos="0 0 0", size="1 1 .01", rgba=".88 .9 .92 1",
                  contype="2", conaffinity="2", condim="3")
    ET.SubElement(world, "camera", name="overview",
                  pos="-.42 -.50 .40", xyaxes=".766 -.643 0 .200 .238 .950")
    pen = ET.SubElement(world, "body", name="pen", pos="0 0 .65")
    ET.SubElement(pen, "freejoint", name="pen_freejoint")
    ET.SubElement(pen, "inertial", pos="0 0 .006875", mass=".016",
                  diaginertia="3.874375e-5 3.874375e-5 2e-7")
    ET.SubElement(pen, "geom", name="pen_collision", type="capsule",
                  size=".005 .060", rgba=".55 .55 .55 0",
                  friction=".6 .001 .0001", condim="3", contype="2",
                  conaffinity="3", priority="1", solref=".02 1.5",
                  solimp=".9 .99 .001", group="3", mass="0")
    for name, position, half_length, color in [
        ("pen_middle_visual", "0 0 0", ".045", ".58 .60 .63 1"),
        ("pen_heavy_visual", "0 0 .055", ".010", ".95 .25 .08 1"),
        ("pen_light_visual", "0 0 -.055", ".010", ".08 .35 .95 1"),
    ]:
        ET.SubElement(pen, "geom", name=name, type="cylinder",
                      size=".005 " + half_length, pos=position, rgba=color,
                      contype="0", conaffinity="0", group="2", mass="0")
    for name, z in [("pen_center", "0"), ("pen_a", ".065"), ("pen_b", "-.065")]:
        ET.SubElement(pen, "site", name=name, pos="0 0 " + z, size=".002",
                      rgba="1 .9 .1 1", group="4")
    sensor = root.find("sensor")
    for sensor_type, name in [
        ("framepos", "pen_position"), ("framequat", "pen_orientation"),
        ("framelinvel", "pen_linvel"), ("frameangvel", "pen_angvel"),
    ]:
        ET.SubElement(sensor, sensor_type, name=name, objtype="body", objname="pen")
    keys = ET.SubElement(root, "keyframe")
    qpos = np.r_[HOME_HAND_QPOS, [0, 0, .65, 1, 0, 0, 0]]
    ET.SubElement(keys, "key", name="home",
                  qpos=" ".join(map(str, qpos)), ctrl=" ".join(map(str, HOME_CTRL)))
    assets = {p.name: p.read_bytes() for p in ASSET_DIR.rglob("*") if p.is_file()}
    return ET.tostring(root, encoding="unicode"), assets


def build_model(config=None) -> mujoco.MjModel:
    scene, assets = scene_and_assets(config)
    return mujoco.MjModel.from_xml_string(scene, assets)


def model_fingerprint(config=None) -> str:
    scene, assets = scene_and_assets(config)
    digest = hashlib.sha256(scene.encode())
    for name, data in sorted(assets.items()):
        digest.update(name.encode()); digest.update(data)
    return digest.hexdigest()


def task_fingerprint(config=None) -> str:
    """Bind a prepared reference to task physics, independent of PPO/budget.

    Changing a task rule requires a schema_version bump. Model geometry and
    upstream assets are additionally bound by model_fingerprint.
    """
    if config is None:
        from .config import PenSpinConfig
        config = PenSpinConfig()
    values = {name: getattr(config, name) for name in TASK_CONFIG_FIELDS}
    if config.task_mode != 'spin':
        values.update(task_mode=config.task_mode,grasp_min_fingers=config.grasp_min_fingers,
                      grasp_acquisition_seconds=config.grasp_acquisition_seconds)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def model_ids(model) -> dict:
    """Stable IDs; finger map is thumb,index,middle,ring,pinky = 0..4."""
    finger = np.full(model.ngeom, -1, dtype=np.int32)
    support = np.zeros(model.ngeom, dtype=bool)
    for g in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[g])).name or ""
        if name in ("right_thumb_proximal_link", "right_thumb_distal_link"):
            finger[g] = 0
        else:
            for index, prefix in enumerate(("right_index", "right_middle", "right_ring", "right_pinky"), 1):
                if name.startswith(prefix):
                    finger[g] = index
        support[g] = name in ("palm", "tetheria_mount", "right_t_link", "right_thumb_mcp_link") or model.geom(g).name == "floor"
    pen_joint = model.joint("pen_freejoint").id
    return {
        "pen_body": model.body("pen").id,
        "pen_geom": model.geom("pen_collision").id,
        "pen_qpos_adr": int(model.jnt_qposadr[pen_joint]),
        "pen_dof_adr": int(model.jnt_dofadr[pen_joint]),
        "site_center": model.site("pen_center").id,
        "site_a": model.site("pen_a").id, "site_b": model.site("pen_b").id,
        "thumb_ip_joint": model.joint("right_thumb_ip").id,
        "thumb_tip_site": model.site("th_tip").id,
        "hand_qpos_ids": np.array([model.jnt_qposadr[model.joint(n).id] for n in HAND_JOINTS]),
        "hand_dof_ids": np.array([model.jnt_dofadr[model.joint(n).id] for n in HAND_JOINTS]),
        "ctrl_lower": model.actuator_ctrlrange[:, 0].copy(),
        "ctrl_upper": model.actuator_ctrlrange[:, 1].copy(),
        "geom_finger": finger, "geom_invalid": support,
    }


def thumb_frame(model, data, ids=None) -> tuple[np.ndarray, np.ndarray]:
    ids = ids or model_ids(model)
    anchor = data.xanchor[ids["thumb_ip_joint"]]
    tip = data.site_xpos[ids["thumb_tip_site"]]
    n = tip - anchor
    return (anchor + tip) * .5, n / np.linalg.norm(n)


def native_contacts(model, data, ids=None) -> dict:
    ids = ids or model_ids(model)
    forces = np.zeros(5)
    invalid = False
    depth = 0.0
    force = np.zeros(6)
    for i, contact in enumerate(data.contact):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        other = g2 if g1 == ids["pen_geom"] else g1 if g2 == ids["pen_geom"] else None
        if other is None:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        normal = max(0.0, float(force[0]))
        group = int(ids["geom_finger"][other])
        if group >= 0:
            forces[group] += normal
        invalid |= bool(ids["geom_invalid"][other] and normal > 1e-6)
        depth = max(depth, -float(contact.dist))
    return {"finger_forces": forces, "invalid_support": invalid, "max_penetration": depth}


def native_pen_velocity(model, data, ids=None):
    ids = ids or model_ids(model)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                            ids["pen_body"], velocity, 0)
    return velocity[3:].copy(), velocity[:3].copy()
