import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test the rough policy on terrains with varying difficulty")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import skrl
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model

# import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.timer import Timer
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.terrains import TerrainGenerator
from omni.isaac.lab.managers import CommandManager

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class UnitreeGo2RoughEnvCfg(InteractiveSceneCfg):
    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.3, use_default_offset=True)
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    # environment
    terrain = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )
    # terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
    # terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
    # terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, policy):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    joint_scale = 1
    past_actions = torch.zeros((scene.num_envs, 12)).to("cuda:0")

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 10000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_state_to_sim(root_state)

            # set joint positions with some noise
            default_joint_pos, default_joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos = torch.rand_like(default_joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, default_joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        velocity_command = torch.tensor([1, 0, 0]).repeat(scene.num_envs, 1).to("cuda:0")

        # get observations
        obs = torch.cat([
            scene["robot"].data.body_state_w[:, 0, 7:10],       # base_lin_vel
            scene["robot"].data.body_state_w[:, 0, 10:],        # base_ang_vel
            scene["robot"].data.projected_gravity_b,            # projected_gravity
            velocity_command,                                   # velocity_commands
            scene["robot"].data.joint_pos - default_joint_pos,  # joint_pos
            scene["robot"].data.joint_vel - default_joint_vel,  # joint_vel
            past_actions,                                       # actions
            # scene["scanner"].data.ray_hits_w[:, :, 2]         # height_scan
        ], dim=1)

        past_actions = actions = default_joint_pos + joint_scale * policy(obs)

        # targets = scene["robot"].data.default_joint_pos  
        scene["robot"].set_joint_position_target(actions)
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print(scene["robot"].data.projected_gravity_b)
        # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        # print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        # print("-------------------------------")
        # print(scene["scanner"])
        # print("Received max height value: ", torch.max(scene["scanner"].data.ray_hits_w[..., -1]).item())
        # print("-------------------------------")

# @configclass
# class CommandsCfg:
#     """Command specifications for the MDP."""

#     base_velocity = mdp.UniformVelocityCommandCfg(
#         asset_name="robot",
#         resampling_time_range=(10.0, 10.0),
#         rel_standing_envs=0.02,
#         rel_heading_envs=1.0,
#         heading_command=True,
#         heading_control_stiffness=0.5,
#         debug_vis=True,
#         ranges=mdp.UniformVelocityCommandCfg.Ranges(
#             lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
#         ),
#     )

def main():
    """Main function."""
    from gymnasium.spaces.box import Box
    import math
    models = {
        "policy": gaussian_model(
            observation_space=Box(-math.inf, math.inf, (48,)),
            action_space=Box(-math.inf, math.inf, (12,)),
            device="cuda:0",
            ml_framework="torch"
        ),
        "value": gaussian_model(
            observation_space=Box(-math.inf, math.inf, (48,)),
            action_space=Box(-math.inf, math.inf, (12,)),
            device="cuda:0",
            ml_framework="torch"
        )
    }

    agent_cfg = PPO_DEFAULT_CONFIG.copy()

    agent_cfg["state_preprocessor_kwargs"].update({"size": Box(-math.inf, math.inf, (48,)), "device": "cuda:0"})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": "cuda:0"})
    agent_cfg["experiment"]["write_interval"] = 0  # don't log to Tensorboard
    agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    agent = PPO(
        models=models,
        memory=None,
        cfg=agent_cfg
    )

    agent.init()
    # load level policy
    policy_path = "models/go2_flat.pt"
    # check if policy file exists
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")

    agent.load(policy_path)

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = UnitreeGo2RoughEnvCfg(num_envs=5, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, policy)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

"""
Actor MLP: Sequential(
  (0): Linear(in_features=235, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=12, bias=True)
)

+----------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (235,)) |
+-----------+--------------------------------+-------------+
|   Index   | Name                           |    Shape    |
+-----------+--------------------------------+-------------+
|     0     | base_lin_vel                   |     (3,)    |
|     1     | base_ang_vel                   |     (3,)    |
|     2     | projected_gravity              |     (3,)    |
|     3     | velocity_commands              |     (3,)    |
|     4     | joint_pos                      |    (12,)    |
|     5     | joint_vel                      |    (12,)    |
|     6     | actions                        |    (12,)    |
|     7     | height_scan                    |    (187,)   |
+-----------+--------------------------------+-------------+
"""