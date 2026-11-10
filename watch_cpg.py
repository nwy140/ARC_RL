"""Watch ARC_RL's scripted CPG walking controller (no trained checkpoint needed)."""
import os
import sys
import time
import argparse
import importlib

if sys.platform == 'win32':
    os.environ['MUJOCO_GL'] = 'glfw'

import gymnasium as gym
import src.envs  # register ARC_RL environments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['bastion', 'leaper', 'queen', 'tick'], default='bastion')
    parser.add_argument('--preset', choices=['slow', 'medium', 'fast'], default='medium')
    args = parser.parse_args()

    mod = importlib.import_module(f'src.controllers.cpg_controller_{args.env}')
    env = gym.make(mod._env_id(args.env), render_mode='human')
    controller = getattr(mod, f'ScriptedCPGController{args.env.capitalize()}')(env, robot=args.env, preset=args.preset)
    try:
        while True:
            mod.reset_env_to_stance(env)
            controller.reset()
            for _ in range(1000):
                action = controller.act()
                _, _, terminated, truncated, _ = env.step(action)
                controller.step_phase()
                if terminated or truncated:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == '__main__':
    main()
