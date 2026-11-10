"""Watch a trained ARC_RL SAC checkpoint in the interactive MuJoCo viewer.
Run from the ARC_RL repository root.
"""
import argparse
import os
import time

if os.name == 'nt':
    os.environ['MUJOCO_GL'] = 'glfw'

import gymnasium as gym
import mujoco.viewer
import torch
import src.envs  # register ARC_RL environments
from src.algos.agent_sac import SACAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['queen', 'bastion', 'leaper', 'tick'], default='queen')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--network-width', type=int, default=256)
    args = parser.parse_args()
    path = args.checkpoint or f'outputs/sac_{args.env}/seed_0/checkpoints/checkpoint_latest.pt'
    if not os.path.isfile(path):
        parser.error(f'Checkpoint not found: {path}')

    env = gym.make(f'{args.env.capitalize()}-v1')
    try:
        agent = SACAgent(env_name=args.env,
                         obs_dim=env.observation_space.shape[0],
                         act_dim=env.action_space.shape[0],
                         act_limit=float(env.action_space.high[0]),
                         device=torch.device('cpu'),
                         hidden_sizes=(args.network_width, args.network_width))
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        agent.policy_net.load_state_dict(ckpt['policy'])
        agent.policy_net.eval()
        print(f'Loaded policy: {path}')
        base = env.unwrapped
        obs, _ = env.reset()
        with mujoco.viewer.launch_passive(base.model, base.data) as viewer:
            while viewer.is_running():
                start = time.perf_counter()
                action = agent.get_test_action(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                viewer.sync()
                if terminated or truncated:
                    obs, _ = env.reset()
                    viewer.sync()
                remaining = base.dt - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        env.close()

if __name__ == '__main__':
    main()
