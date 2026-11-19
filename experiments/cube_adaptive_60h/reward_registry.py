"""Operational registry: leave the historical, fingerprinted shaping intact."""
from .shaping import TRANSFORMS as LEGACY_TRANSFORMS
from .thumb_drive import TRANSFORM as THUMB_DRIVE

TRANSFORMS = (*LEGACY_TRANSFORMS, THUMB_DRIVE)


def wrap(env, transform):
    from .shaping import NarrowGoalReward, MountMarginReward
    if transform not in TRANSFORMS:
        raise ValueError('Unknown training reward transform')
    if transform:
        env = NarrowGoalReward(env)
    if transform in ('narrow_small_angle_mount_margin_v1', 'center_missing_tip_mount_v1', THUMB_DRIVE):
        env = MountMarginReward(env)
    if transform in ('center_missing_tip_mount_v1', THUMB_DRIVE):
        from .capture_geometry import CaptureGeometryReward
        env = CaptureGeometryReward(env)
    if transform == THUMB_DRIVE:
        from .thumb_drive import ThumbDriveReward
        env = ThumbDriveReward(env)
    return env
