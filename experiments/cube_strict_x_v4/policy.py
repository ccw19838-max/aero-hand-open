"""Extend the matched V3 policy with six causally computed support features."""
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from experiments.cube_adaptive_60h.policy import Extractor, Policy as V3Policy
from experiments.cube_official_baseline.policy import mlp


class Features(BaseFeaturesExtractor):
    def __init__(self, space):
        if space.shape != (149,):
            raise ValueError('Expected 149-dimensional observations')
        super().__init__(space, features_dim=149)
        self.register_buffer('mean', torch.zeros(149))
        self.register_buffer('std', torch.ones(149))

    def forward(self, observation):
        return (observation - self.mean) / self.std.clamp_min(1e-6)


class WideExtractor(Extractor):
    def __init__(self):
        nn.Module.__init__(self)
        self.policy_net = mlp(149)
        self.value_net = mlp(149)
        self.latent_dim_pi = self.latent_dim_vf = 128


class Policy(V3Policy):
    def __init__(self, *args, **kwargs):
        kwargs.update(features_extractor_class=Features, ortho_init=False)
        ActorCriticPolicy.__init__(self, *args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = WideExtractor().to(self.device)


@torch.no_grad()
def initialize_from_v3(dst_policy, src_policy):
    """Copy initialized V3 Gaussian weights, including action/value heads and log std."""
    old = src_policy.state_dict()
    for name, parameter in dst_policy.state_dict().items():
        previous = old[name]
        if previous.shape == parameter.shape:
            parameter.copy_(previous)
        elif parameter.ndim == 2 and parameter.shape[1] == 149 and previous.shape == (parameter.shape[0], 143):
            parameter.zero_()
            parameter[:, :143].copy_(previous)
        elif parameter.shape == (149,) and previous.shape == (143,) and name.endswith(('.mean', '.std')):
            parameter.fill_(1. if name.endswith('.std') else 0.)
            parameter[:143].copy_(previous)
        else:
            raise ValueError(f'Incompatible transfer for {name}: {previous.shape} to {parameter.shape}')
