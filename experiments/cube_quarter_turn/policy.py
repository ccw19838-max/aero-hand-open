"""Goal observations extend the existing manipulation prior, with explicit calibration."""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from experiments.cube_official_baseline.policy import SourceFeatures,OfficialPolicy,mlp


class GoalFeatures(SourceFeatures):
    def __init__(self,observation_space):
        super().__init__(observation_space);self._features_dim=157
    def forward(self,observations):
        base=super().forward(observations);goal=observations['goal_state']
        return torch.cat((base[...,:14],goal,base[...,14:],goal),dim=-1)


class GoalExtractor(nn.Module):
    def __init__(self):
        super().__init__();self.policy_net=mlp(45);self.value_net=mlp(112)
        self.latent_dim_pi=128;self.latent_dim_vf=128
    def forward_actor(self,features):return self.policy_net(features[...,:45])
    def forward_critic(self,features):return self.value_net(features[...,45:])
    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)


class QuarterPolicy(OfficialPolicy):
    def __init__(self,*args,**kwargs):
        kwargs.update(features_extractor_class=GoalFeatures,ortho_init=False)
        ActorCriticPolicy.__init__(self,*args,**kwargs)
    def _build_mlp_extractor(self):self.mlp_extractor=GoalExtractor().to(self.device)


def initialize_prior(model,source_model,reference_observation,calibrate=True):
    """Copy hidden features, add zero goal columns, reset value for the new reward.

    Absolute motor targets are now centred on the approved 57-mm reference.
    Subtracting the reference logit gives neutral deterministic actions there,
    while preserving the prior's response to changes in feedback.
    """
    target=model.policy;source=source_model.policy
    with torch.no_grad():
        for key in ('state','privileged_state'):
            for stat in ('mean','std'):
                getattr(target.features_extractor,key+'_'+stat).copy_(getattr(source.features_extractor,key+'_'+stat))
        for kind,size in [('policy_net',14),('value_net',81)]:
            dst=getattr(target.mlp_extractor,kind);src=getattr(source.mlp_extractor,kind)
            dst[0].weight.zero_();dst[0].weight[:,:size].copy_(src[0].weight);dst[0].bias.copy_(src[0].bias)
            for index in (2,4):dst[index].load_state_dict(src[index].state_dict())
        target.action_net.load_state_dict(source.action_net.state_dict())
        offset=np.zeros(7)
        if calibrate:
            raw={k:reference_observation[k] for k in ('state','privileged_state')}
            tensors=source.obs_to_tensor(raw)[0]
            features=source.features_extractor(tensors)
            offset=source.action_net[0](source.mlp_extractor.forward_actor(features)).cpu().numpy().reshape(-1)
            target.action_net[0].bias.sub_(torch.as_tensor(offset,device=target.device))
        target.value_net.weight.zero_();target.value_net.bias.zero_()
        target.log_std.fill_(float(np.log(.3)))
    return dict(actor_inputs=45,critic_inputs=112,goal_features=31,
        added_goal_weights='zero initialized',reference_logit_offset=offset.tolist(),
        reference_calibration=calibrate,critic_output='reset for the new endpoint/holding reward',
        semantics='Copied manipulation prior features and actor output, zero-padded goal inputs; calibrated neutral action at prepared reference; fresh optimizer, critic output and exploration.')
