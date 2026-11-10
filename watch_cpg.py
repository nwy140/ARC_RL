
"""Watch ARC_RL robots walk using their built-in scripted CPG controllers.

No trained checkpoint, Hugging Face download, or dataset collection is required.
Run these commands from the ARC_RL project folder with the godotmujoco environment active.

QUICK START (PowerShell / CMD):
    conda activate godotmujoco
    python watch_cpg.py --env bastion

EXAMPLES — ALL ROBOTS:
    python watch_cpg.py --env bastion
    python watch_cpg.py --env leaper
    python watch_cpg.py --env queen
    python watch_cpg.py --env tick

EXAMPLES — PLAYBACK SPEED (simulation time relative to wall-clock time):
    python watch_cpg.py --env leaper --speed 0.25   # Quarter speed
    python watch_cpg.py --env leaper --speed 0.5    # Half speed
    python watch_cpg.py --env leaper --speed 1      # Real time (default)
    python watch_cpg.py --env leaper --speed 2      # Double speed

EXAMPLES — WALKING GAIT PRESETS (independent of playback speed):
    python watch_cpg.py --env bastion --preset slow
    python watch_cpg.py --env queen --preset medium
    python watch_cpg.py --env tick --preset fast
    python watch_cpg.py --env leaper --preset slow --speed 0.5

HELP:
    python watch_cpg.py --help

Press Ctrl+C to stop. This is a scripted walking demonstration, NOT a
trained reinforcement-learning policy or a playback of the HDF5 dataset.
"""

import os
import sys
import time
import argparse
import importlib

# Windows does not support MuJoCo's Linux EGL rendering backend.
if sys.platform == "win32":
    os.environ["MUJOCO_GL"] = "glfw"

import gymnasium as gym
import src.envs  # Register the custom ARC_RL Gymnasium environment IDs.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        choices=["bastion", "leaper", "queen", "tick"],
        default="bastion",
    )

    parser.add_argument(
        "--preset",
        choices=["slow", "medium", "fast"],
        default="medium",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed: 0.5=half, 1=real time, 2=double",
    )

    args = parser.parse_args()

    if not (0 < args.speed <= 100):
        parser.error("--speed must be greater than 0 and at most 100")

    # Load the robot-specific CPG controller.
    # Example: src.controllers.cpg_controller_leaper
    mod = importlib.import_module(
        f"src.controllers.cpg_controller_{args.env}"
    )

    # Human render mode opens MuJoCo's on-screen viewer.
    env = gym.make(
        mod._env_id(args.env),
        render_mode="human",
    )

    # The scripted controller outputs joint actions.
    # No trained .pt checkpoint is needed.
    controller = getattr(
        mod,
        f"ScriptedCPGController{args.env.capitalize()}",
    )(
        env,
        robot=args.env,
        preset=args.preset,
    )

    # Gymnasium MuJoCo env.dt includes frame_skip.
    # It represents the simulated time advanced by each env.step().
    step_duration = float(env.unwrapped.dt) / args.speed

    next_step_time = time.perf_counter()

    print(
        f"Watching {args.env} at {args.speed:g}x playback "
        "(Ctrl+C to stop)"
    )

    try:
        while True:
            # Not every robot controller exposes the same reset helper.
            if hasattr(mod, "reset_env_to_stance"):
                mod.reset_env_to_stance(env)
            else:
                env.reset()

            controller.reset()

            for _ in range(1000):
                # Generate the next action from the walking controller.
                action = controller.act()

                # Advance the physics simulation.
                _, _, terminated, truncated, _ = env.step(action)

                # Advance the CPG walking phase.
                controller.step_phase()

                # Pace simulation against real time.
                # --speed 0.5 = half speed
                # --speed 1   = real time
                # --speed 2   = double speed
                next_step_time += step_duration

                delay = next_step_time - time.perf_counter()

                if delay > 0:
                    time.sleep(delay)
                else:
                    # Avoid accumulating timing backlog when rendering
                    # takes longer than the target frame duration.
                    next_step_time = time.perf_counter()

                # Restart the episode when the environment finishes.
                if terminated or truncated:
                    break

    except KeyboardInterrupt:
        pass

    finally:
        # Cleanly close the MuJoCo viewer.
        env.close()


if __name__ == "__main__":
    main()
