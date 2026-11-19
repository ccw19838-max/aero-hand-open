from dataclasses import dataclass,asdict
from experiments.cube_contact_ppo.task import Config as BaseConfig


@dataclass(frozen=True)
class Profile:
    name: str = 'contact_recovery_v1'
    learning_rate: float = 1e-4
    gamma: float = .997
    entropy: float = .002
    action_fraction: float = .02
    contact_weight: float = .6
    loss_weight: float = .8
    loss_grace: float = .35
    pressure_potential_gain: float = 2.
    stall_weight: float = .8
    progress_gain: float = 25.
    grip_mix: float = .15
    bc_weight: float = .1
    def dictionary(self):return asdict(self)


@dataclass(frozen=True)
class Config(BaseConfig):
    version: str = 'adaptive_xy_60h_v1'
    stagnation_seconds: float = 8.


LADDER=(0.,10.,20.,30.,45.,60.,75.,90.)


def validate_profile(profile):
    p=profile.dictionary()
    bounds=dict(learning_rate=(2e-5,3e-4),gamma=(.99,.9995),entropy=(.0005,.01),
        action_fraction=(.005,.03),contact_weight=(0.,3.),loss_weight=(0.,3.),loss_grace=(.15,1.),
        pressure_potential_gain=(0.,5.),stall_weight=(0.,3.),progress_gain=(20.,50.),
        grip_mix=(0.,.35),bc_weight=(0.,.3))
    for key,(lower,upper) in bounds.items():
        if not lower<=p[key]<=upper:raise ValueError('Profile outside approved engineering bounds: '+key)
    return profile
