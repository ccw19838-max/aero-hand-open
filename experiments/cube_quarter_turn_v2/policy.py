"""Full-state teacher warm start; new optimizer, control semantics and reward."""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from experiments.cube_official_baseline.policy import OfficialPolicy, mlp
from .env import OBS_SIZE


class TeacherFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        super().__init__(observation_space,features_dim=OBS_SIZE)
        self.register_buffer('mean',torch.zeros(OBS_SIZE))
        self.register_buffer('std',torch.ones(OBS_SIZE))
    def forward(self,obs): return (obs-self.mean)/self.std.clamp_min(1e-6)


class TeacherExtractor(nn.Module):
    def __init__(self):
        super().__init__();self.policy_net=mlp(OBS_SIZE);self.value_net=mlp(OBS_SIZE)
        self.latent_dim_pi=self.latent_dim_vf=128
    def forward_actor(self,x): return self.policy_net(x)
    def forward_critic(self,x): return self.value_net(x)
    def forward(self,x): return self.forward_actor(x),self.forward_critic(x)


class TeacherPolicy(OfficialPolicy):
    def __init__(self,*args,**kwargs):
        kwargs.update(features_extractor_class=TeacherFeatures,ortho_init=False)
        ActorCriticPolicy.__init__(self,*args,**kwargs)
    def _build_mlp_extractor(self): self.mlp_extractor=TeacherExtractor().to(self.device)


def initialize_prior(model,source,reference_obs):
    target=model.policy; old=source.policy
    with torch.no_grad():
        target.features_extractor.mean[:81].copy_(old.features_extractor.privileged_state_mean)
        target.features_extractor.std[:81].copy_(old.features_extractor.privileged_state_std)
        target.features_extractor.mean[:14].copy_(old.features_extractor.state_mean)
        target.features_extractor.std[:14].copy_(old.features_extractor.state_std)
        for kind,size in (('policy_net',14),('value_net',81)):
            dst=getattr(target.mlp_extractor,kind);src=getattr(old.mlp_extractor,kind)
            dst[0].weight.zero_();dst[0].weight[:,:size].copy_(src[0].weight);dst[0].bias.copy_(src[0].bias)
            for index in (2,4): dst[index].load_state_dict(src[index].state_dict())
        target.action_net.load_state_dict(old.action_net.state_dict())
        tensor=target.obs_to_tensor(reference_obs)[0]
        logit=target.action_net[0](target.mlp_extractor.forward_actor(target.features_extractor(tensor)))
        target.action_net[0].bias.sub_(logit.reshape(-1))
        target.value_net.weight.zero_();target.value_net.bias.zero_()
        target.log_std.fill_(float(np.log(.15)))
    return {'actor_inputs':OBS_SIZE,'critic_inputs':OBS_SIZE,'neutral_calibration':True,
        'semantics':'feature warm start only; changed action range, previous-action history and task; fresh optimizer/value head'}


def cloning_update(model,observations,actions,indices,weight=1.):
    tensor=torch.as_tensor(observations[indices],device=model.device)
    expected=torch.as_tensor(actions[indices],device=model.device)
    policy=model.policy
    latent=policy.mlp_extractor.forward_actor(policy.features_extractor(tensor))
    loss=weight*torch.mean((policy.action_net(latent)-expected)**2)
    policy.optimizer.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);policy.optimizer.step()
    return float(loss.detach().cpu())
