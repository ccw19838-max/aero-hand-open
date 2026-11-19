import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from experiments.cube_official_baseline.policy import OfficialPolicy,mlp
from .env import OBS_SIZE


class Features(BaseFeaturesExtractor):
    def __init__(self,space):
        super().__init__(space,features_dim=OBS_SIZE)
        self.register_buffer('mean',torch.zeros(OBS_SIZE));self.register_buffer('std',torch.ones(OBS_SIZE))
    def forward(self,x):return (x-self.mean)/self.std.clamp_min(1e-6)


class Extractor(nn.Module):
    def __init__(self):
        super().__init__();self.policy_net=mlp(OBS_SIZE);self.value_net=mlp(OBS_SIZE)
        self.latent_dim_pi=self.latent_dim_vf=128
    def forward_actor(self,x):return self.policy_net(x)
    def forward_critic(self,x):return self.value_net(x)
    def forward(self,x):return self.forward_actor(x),self.forward_critic(x)


class Policy(OfficialPolicy):
    def __init__(self,*args,**kwargs):
        kwargs.update(features_extractor_class=Features,ortho_init=False)
        ActorCriticPolicy.__init__(self,*args,**kwargs)
    def _build_mlp_extractor(self):self.mlp_extractor=Extractor().to(self.device)


def transfer(model,source):
    target=model.policy;old=source.policy;size=old.features_extractor.mean.numel()
    if size not in (138,143):raise ValueError('Unsupported transfer observation semantics')
    with torch.no_grad():
        target.features_extractor.mean[:size].copy_(old.features_extractor.mean)
        target.features_extractor.std[:size].copy_(old.features_extractor.std)
        for kind in ('policy_net','value_net'):
            dst=getattr(target.mlp_extractor,kind);src=getattr(old.mlp_extractor,kind)
            dst[0].weight.zero_();dst[0].weight[:,:size].copy_(src[0].weight);dst[0].bias.copy_(src[0].bias)
            for index in (2,4):dst[index].load_state_dict(src[index].state_dict())
        target.action_net.load_state_dict(old.action_net.state_dict())
        target.log_std.copy_(old.log_std.clamp(np.log(.08),np.log(.25)))
        target.value_net.weight.zero_();target.value_net.bias.zero_()
    return dict(source_observations=size,new_observations=143,actor_transferred=True,
        new_columns='zero',critic_output='reset for new reward',optimizer='fresh',bitwise_resume=False)


class RetainPPO(PPO):
    def train(self):
        super().train()
        if getattr(self,'retention',None) is not None:
            obs,actions,weight,rng=self.retention
            policy=self.policy
            for _ in range(2):
                indices=rng.integers(len(actions),size=128)
                x=torch.as_tensor(obs[indices],device=self.device);expected=torch.as_tensor(actions[indices],device=self.device)
                latent=policy.mlp_extractor.forward_actor(policy.features_extractor(x))
                loss=weight*torch.mean((policy.action_net(latent)-expected)**2)
                policy.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);policy.optimizer.step()
            self.logger.record('train/retention_loss',float(loss.detach()))
    def _excluded_save_params(self):return super()._excluded_save_params()+['retention']
