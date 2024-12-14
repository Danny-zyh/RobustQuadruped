import math
import torch

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import TerminationTermCfg as TerminationTerm
from omni.isaac.lab_tasks.manager_based.locomotion.velocity import mdp
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip

from .flat_env_cfg import UnitreeGo2FlatEnvCfg
from ...velocity_env_cfg import MySceneCfg, CommandsCfg, ActionsCfg, ObservationsCfg, RewardsCfg, TerminationsCfg


@configclass
class UnitreeGo2FlatBaseCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # task related reward
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # set unrelated penalty reward to 0
        self.rewards.feet_air_time.weight = 0
        self.rewards.undesired_contacts = None
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0


class ExpObservationCfg(ObservationsCfg):
    @configclass
    class JointPos(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    
    joint_position: JointPos = JointPos()


@configclass
class UnitreeGo2FlatBaseExpCfg(UnitreeGo2FlatEnvCfg):
    observations: ExpObservationCfg = ExpObservationCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # task related reward
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # set unrelated penalty reward to 0
        self.rewards.feet_air_time.weight = 0
        self.rewards.undesired_contacts = None
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0


@configclass
class RandForceEventCfg:
    # random_joint_gain = EventTerm(
    #     func=mdp.events.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "stiffness_distribution_params": (50, 100),
    #         "damping_distribution_params": (50, 100)
    #     }
    # )

    random_external_force = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-1.5, 1.5),
            "torque_range": (-0.5, 0.5)
        },
        interval_range_s=(5, 10)
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
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
    )


@configclass
class UnitreeGo2FlatRandForceCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: RandForceEventCfg = RandForceEventCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # task related reward
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # set unrelated penalty reward to 0
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0


@configclass
class TestEventCfg:
    # random_joint_gain = EventTerm(
    #     func=mdp.events.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "stiffness_distribution_params": (50, 100),
    #         "damping_distribution_params": (50, 100)
    #     }
    # )
    random_external_force = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10, 10),
            "torque_range": (-5, 5)
        },
        interval_range_s=(5, 10)
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-3, 3), "y": (-3, 3)}},
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
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
    )

    random_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.7),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
    )


def check_fall(env: ManagerBasedEnv):
    return mdp.projected_gravity(env=env)[:, 2] > 0.2


@configclass
class TestTerminationCfg:
    check_fall = TerminationTerm(func=check_fall)


@configclass
class UnitreeGo2FlatTestCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=500, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TestTerminationCfg = TestTerminationCfg()
    events: TestEventCfg = TestEventCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # task related reward
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # set unrelated penalty reward to 0
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0
