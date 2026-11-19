"""Run with the reference JAX/Brax venv to export the trusted bundled weights."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();params=model.load_params(args.weights);output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True);arrays={}
    for key in ('state','privileged_state'):
        for stat in ('mean','std'):arrays[key+'_'+stat]=np.asarray(getattr(params[0],stat)[key])
    for prefix,component in [('actor',params[1]),('critic',params[2])]:
        for i,layer in enumerate(component['params'].values()):
            for key,value in layer.items():arrays[f'{prefix}_{i}_{key}']=np.asarray(value)
    rng=np.random.default_rng(324)
    obs={key:(arrays[key+'_mean']+arrays[key+'_std']*rng.normal(size=(64,size))).astype(np.float32)
         for key,size in [('state',14),('privileged_state',81)]}
    net=networks.make_ppo_networks({'state':14,'privileged_state':81},7,
        preprocess_observations_fn=running_statistics.normalize,policy_hidden_layer_sizes=(512,256,128),
        value_hidden_layer_sizes=(512,256,128),policy_obs_key='state',value_obs_key='privileged_state')
    inference=jax.jit(networks.make_inference_fn(net)(params,deterministic=True))
    arrays['test_state']=obs['state'];arrays['test_privileged_state']=obs['privileged_state']
    arrays['test_actions']=np.asarray(inference(obs,jax.random.PRNGKey(0))[0])
    arrays['test_values']=np.asarray(net.value_network.apply(params[0],params[2],obs))
    np.savez_compressed(output,**arrays)
    report=dict(source_weights=str(Path(args.weights).resolve()),
        source_sha256=hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        converted_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        actor_observations=14,critic_observations=81,hidden_layers=[512,256,128],
        deterministic_test_observations=64,stochastic_distribution_restored=False)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
