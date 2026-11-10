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
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed: 0.5=half, 1=real time, 2=double')
    args = parser.parse_args()
    if not (0 < args.speed <= 100):
        parser.error('--speed must be greater than 0 and at most 100')

    mod = importlib.import_module(f'src.controllers.cpg_controller_{args.env}')
    env = gym.make(mod._env_id(args.env), render_mode='human')
    controller = getattr(mod, f'ScriptedCPGController{args.env.capitalize()}')(env, robot=args.env, preset=args.preset)
    # Gymnasium MuJoCo env.dt includes frame_skip; unlike model.opt.timestep,
    # it represents the simulated time advanced by each env.step(action).
    step_duration = float(env.unwrapped.dt) / args.speed
    next_step_time = time.perf_counter()
    print(f"Watching {args.env} at {args.speed:g}x playback (Ctrl+C to stop)")
    try:
        while True:
            mod.reset_env_to_stance(env)
            controller.reset()
            for _ in range(1000):
                action = controller.act()
                _, _, terminated, truncated, _ = env.step(action)
                controller.step_phase()
                next_step_time += step_duration
                delay = next_step_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    # If the renderer is too slow, avoid accumulating a backlog.
                    next_step_time = time.perf_counter()
                if terminated or truncated:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == '__main__':
    main()
