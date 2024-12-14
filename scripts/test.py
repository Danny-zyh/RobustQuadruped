"""
Script to test the trained agent. Result evaluates mean fall
time and cumulative task reward
"""


import argparse
from omni.isaac.lab.app import AppLauncher

# Launch Isaac Sim
# add argparse arguments
parser = argparse.ArgumentParser(description="Test a checkpoint of an RL agent from skrl.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.", required=True)
parser.add_argument("--output", type=str, default=None, help="Path to output directory", required=True)
parser.add_argument("--max_timestep", type=int, default=10000, help="Maximum timestep to simulate", required=False)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import os
import torch
import json
import numpy as np

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model

from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg


def main():
    # parse save dir
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    output_path = os.path.abspath(args_cli.output)
    resume_path = os.path.abspath(args_cli.checkpoint)
    
    # config environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # config model for ppo
    if experiment_cfg["models"]["separate"]:
        models = {
            "policy": gaussian_model(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                **process_skrl_cfg(experiment_cfg["models"]["policy"], ml_framework="torch"),
            ),
            "value": deterministic_model(  # critic
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                **process_skrl_cfg(experiment_cfg["models"]["value"], ml_framework="torch"),
            )
        }
    else:
        models = {
            "policy": shared_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                process_skrl_cfg(experiment_cfg["models"]["policy"], ml_framework="torch"),
                process_skrl_cfg(experiment_cfg["models"]["value"], ml_framework="torch"),
            ],
        )}
        models["value"] = models["policy"]


    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"], ml_framework="torch"))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    agent_cfg["experiment"]["write_interval"] = 0  # don't log to Tensorboard
    agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    agent = PPO(
        models=models,
        memory=None,  # memory is optional during evaluation
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    agent.init()
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.load(resume_path)
    agent.set_running_mode("eval")  # set agent to evaluation mode

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # initialize the fall time and cumulative reward calculation
    fall_time = -1*torch.ones(env.scene.num_envs).to(env.device)
    cumulative_reward = torch.zeros(env.scene.num_envs).to(env.device)

    joint_positions = []

    running = lambda: fall_time == -1
    
    # simulate environment
    while any(running()) and timestep < args_cli.max_timestep \
        and simulation_app.is_running():
        with torch.inference_mode():
            actions = agent.act(obs, timestep=0, timesteps=0)[0]
            obs, _, _, _, _ = env.step(actions)

            # record the fall time of agents that just fall
            terminated = env.unwrapped.termination_manager.terminated
            fall_time[terminated & running()] = timestep

            # add reward
            cumulative_reward[running()] += env.unwrapped.reward_manager.\
                compute(0.005)[running()]
            
            # record joint position
            joint_positions.append(env.unwrapped.observation_manager.compute_group("joint_position").\
                                  detach().to("cpu").numpy())

            timestep += 1
    
    # close the simulator
    env.close()

    results = {
        "fall_time": fall_time.cpu().tolist(),
        "cumulative_reward": cumulative_reward.cpu().tolist()
    }

    # Write results to a JSON file
    results_file = os.path.join(f"{output_path}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    joint_position_file = os.path.join("/tmp/joint_positions.json")
    np.save(joint_position_file, np.array(joint_positions))

    print(f"[INFO] Evaluation results saved to: {results_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import sys, traceback
        print(f"[ERROR]: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        simulation_app.close()
