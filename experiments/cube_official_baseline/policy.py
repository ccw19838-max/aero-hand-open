"""Asymmetric 14-input actor / 81-input critic with legacy normalization buffers.

The deterministic actor exactly matches the converted Brax SiLU MLP and tanh
output. Fine tuning uses SB3's clipped Gaussian exploration, not Brax's full
state-dependent tanh-normal distribution; it is not a bitwise Brax continuation.
"""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SourceFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        super().__init__(observation_space,features_dim=95)
        for key,size in [('state',14),('privileged_state',81)]:
            self.register_buffer(key+'_mean',torch.zeros(size))
            self.register_buffer(key+'_std',torch.ones(size))
    def forward(self,observations):
        return torch.cat([(observations[k]-getattr(self,k+'_mean'))/getattr(self,k+'_std')
                          for k in ('state','privileged_state')],dim=-1)


def mlp(size):
    layers=[]
    for units in (512,256,128):layers.extend([nn.Linear(size,units),nn.SiLU()]);size=units
    return nn.Sequential(*layers)


class AsymmetricExtractor(nn.Module):
    def __init__(self):
        super().__init__();self.policy_net=mlp(14);self.value_net=mlp(81)
        self.latent_dim_pi=128;self.latent_dim_vf=128
    def forward_actor(self,features):return self.policy_net(features[...,:14])
    def forward_critic(self,features):return self.value_net(features[...,14:])
    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)


class OfficialPolicy(ActorCriticPolicy):
    def __init__(self,*args,**kwargs):
        kwargs.update(features_extractor_class=SourceFeatures,ortho_init=False)
        super().__init__(*args,**kwargs)
    def _build_mlp_extractor(self):self.mlp_extractor=AsymmetricExtractor().to(self.device)
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        # Reuse the same Linear parameters already owned by the optimizer.
        self.action_net=nn.Sequential(self.action_net,nn.Tanh())


def initialize_source(model,archive):
    with np.load(archive,allow_pickle=False) as data:
        with torch.no_grad():
            features=model.policy.features_extractor
            for key in ('state','privileged_state'):
                for stat in ('mean','std'):
                    getattr(features,key+'_'+stat).copy_(torch.as_tensor(data[key+'_'+stat]))
            for prefix,network in [('actor',model.policy.mlp_extractor.policy_net),
                                   ('critic',model.policy.mlp_extractor.value_net)]:
                for i in range(3):
                    network[2*i].weight.copy_(torch.as_tensor(data[f'{prefix}_{i}_kernel'].T))
                    network[2*i].bias.copy_(torch.as_tensor(data[f'{prefix}_{i}_bias']))
            model.policy.action_net[0].weight.copy_(torch.as_tensor(data['actor_3_kernel'][:,:7].T))
            model.policy.action_net[0].bias.copy_(torch.as_tensor(data['actor_3_bias'][:7]))
            model.policy.value_net.weight.copy_(torch.as_tensor(data['critic_3_kernel'].T))
            model.policy.value_net.bias.copy_(torch.as_tensor(data['critic_3_bias']))
            model.policy.log_std.fill_(float(np.log(.15)))
