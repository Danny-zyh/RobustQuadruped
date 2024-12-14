import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Creating a quadruped base environment.")
# Create a new argument group
model_group = parser.add_argument_group('Model Arguments')
model_group.add_argument('model_path_pt', type=str, help='Path to the model file (e.g., model_path.pt)')

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import math
import torch

from rsl_rl.modules import ActorCritic

# import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.managers import TerminationTermCfg, TerminationManager
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  


import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class UnitreeGo2SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1,
            dynamic_friction=1,
            restitution=1
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class UnitreeGo2ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.3, use_default_offset=True)


@configclass
class UnitreeGo2ObservationCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class UnitreeGo2EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="interval",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.8),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
        interval_range_s=(5.0, 7.0)
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-4.0, 4.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="interval",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
        interval_range_s=(5.0, 7.0)
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-1, 1), "y": (-1, 1)}},
    )

def check_fall(env: ManagerBasedEnv):
    return mdp.projected_gravity(env=env)[:, 2] > 0.2

@configclass
class UnitreeGo2TerminationCfg:
    check_fall = TerminationTermCfg(func=check_fall)

@configclass
class UnitreeGo2EnvCfg(ManagerBasedRLEnvCfg):
    # commands = CommandsCfg()
    scene = UnitreeGo2SceneCfg(num_envs=300, env_spacing=2.5)
    observations = UnitreeGo2ObservationCfg()
    actions = UnitreeGo2ActionsCfg()
    events = UnitreeGo2EventCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations = UnitreeGo2TerminationCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material

def main(policy_path):
    """Main function."""
    # setup base environment
    env_cfg = UnitreeGo2EnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # load level policy
    # check if policy file exists
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")

    loaded_dict = torch.load(policy_path)
    actor_critic = ActorCritic(48, 48, 12, [128]*3, [128]*3).to("cuda:0")
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])

    policy = actor_critic.act_inference

    # simulate physics
    count = 0
    obs, _ = env.reset()
    fall_time = -1*torch.ones(env.scene.num_envs).to("cuda:0")

    obs, _ = env.reset()
    while simulation_app.is_running() and any(fall_time.flatten() == -1) and count < 10000:
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                print(f"[INFO]: {sum(fall_time != -1)} falls")
            action = policy(obs["policy"])

            # record failure time
            fall_time[env.termination_manager.terminated 
                        & (fall_time == -1)] = count
            # step env
            obs, *_ = env.step(action)
            count += 1

    # close the environment
    env.close()
    return fall_time

import json
import argparse

if __name__ == "__main__":


    # Access the model path argument
    model_path = args_cli.model_path_pt
    print("model path is", model_path)
    fall_time = main(model_path).tolist()

    print("fall_time", fall_time)

    with open(f"../evals/{model_path.split('model_')[-1][:-3]}.json", "w") as fp:
        json.dump(fall_time, fp)

    simulation_app.close()
