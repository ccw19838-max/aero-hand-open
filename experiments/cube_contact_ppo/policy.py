"""Small full-state CPU policy; feature warm start, new incremental action head."""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from experiments.cube_official_baseline.policy import OfficialPolicy,mlp
from experiments.cube_quarter_turn_v2.policy import initialize_prior as initialize_features
from .env import OBS_SIZE


class Features(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        super().__init__(observation_space,features_dim=OBS_SIZE)
        self.register_buffer('mean',torch.zeros(OBS_SIZE));self.register_buffer('std',torch.ones(OBS_SIZE))
    def forward(self,obs):return (obs-self.mean)/self.std.clamp_min(1e-6)


class Extractor(nn.Module):
    def __init__(self):
        super().__init__();self.policy_net=mlp(OBS_SIZE);self.value_net=mlp(OBS_SIZE)
        self.latent_dim_pi=self.latent_dim_vf=128
    def forward_actor(self,features):return self.policy_net(features)
    def forward_critic(self,features):return self.value_net(features)
    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)


class Policy(OfficialPolicy):
    def __init__(self,*args,**kwargs):
        kwargs.update(features_extractor_class=Features,ortho_init=False)
        ActorCriticPolicy.__init__(self,*args,**kwargs)
    def _build_mlp_extractor(self):self.mlp_extractor=Extractor().to(self.device)


def initialize(model,source,observation):
    provenance=initialize_features(model,source,observation)
    with torch.no_grad():
        # Source outputs were absolute targets for continuous spin. Reusing that
        # head for accumulated increments would integrate a spurious bias.
        model.policy.action_net[0].weight.zero_();model.policy.action_net[0].bias.zero_()
        model.policy.log_std.fill_(float(np.log(.25)))
    provenance.update(actor_inputs=OBS_SIZE,critic_inputs=OBS_SIZE,
        action_head='reset to neutral increments; source hidden features/normalizers retained',
        optimizer='fresh PPO optimizer',demonstrations_used=False,
        architecture='full-state 512/256/128 SiLU MLP; not the 2019 LSTM architecture')
    return provenance
