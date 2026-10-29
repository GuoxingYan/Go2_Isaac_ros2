from isaaclab.scene import InteractiveSceneCfg
from isaaclab_assets.robots.spot import SPOT_CFG

from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg
from isaacsim.core.utils.viewports import set_camera_view
import numpy as np
from scipy.spatial.transform import Rotation as R
import quadruped.spot.spot_ctrl as spot_ctrl


@configclass
class SpotSimCfg(InteractiveSceneCfg):
    """声明 Spot 的共享地面、灯光、关节系统与足端接触传感器。"""
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0, 0, 1e-4)
        )
    )
    
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/DomeLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    # )

    # Spot Robot
    spot: ArticulationCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/spot")
    # 环境命名空间会展开为多个 env_i；场景字典中的资产键为 spot。
    
    # Spot foot contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/spot/.*_foot", history_length=3, track_air_time=True)

    # Spot height scanner
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/spot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20)), 
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), 
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    # 策略动作是关节位置控制量，不是直接修改机身位置；执行器和物理引擎决定实际运动。
    joint_pos = mdp.JointPositionActionCfg(asset_name="spot", joint_names=[".*"])

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # 状态、重力方向、目标速度、关节状态和上一次动作的顺序必须保持训练时约定。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
                               params={"asset_cfg": SceneEntityCfg(name="spot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,
                               params={"asset_cfg": SceneEntityCfg(name="spot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    params={"asset_cfg": SceneEntityCfg(name="spot")},
                                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
        # velocity command
        base_vel_cmd = ObsTerm(func=spot_ctrl.base_vel_cmd)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg(name="spot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                            params={"asset_cfg": SceneEntityCfg(name="spot")})
        actions = ObsTerm(func=mdp.last_action)
        
        # Height scan
        height_scan = ObsTerm(func=mdp.height_scan,
                              params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                              clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            """关闭噪声并把观测项拼成单个策略输入，保持推理输入可重复。"""
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # 实际速度目标由 spot_ctrl 的张量提供，此零范围命令管理器不是遥控输入来源。
    base_vel_cmd = mdp.UniformVelocityCommandCfg(
        asset_name="spot",
        resampling_time_range=(0.0, 0.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )

@configclass
class EventCfg:
    """Configuration for events."""
    pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass



@configclass
class SpotRSLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Spot environment."""
    # scene settings
    scene = SpotSimCfg(num_envs=1, env_spacing=2.0)

    # basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    
    # dummy settings
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        """配置 200 Hz 物理步、初始视角和动作尺度；main.py 再决定控制分频。"""
        # viewer settings
        self.viewer.eye = [-4.0, 0.0, 5.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        # step settings
        self.decimation = 8  # step

        # simulation settings
        self.sim.dt = 0.005  # sim step every 
        self.sim.render_interval = self.decimation  
        self.sim.disable_contact_processing = True
        self.sim.render.antialiasing_mode = None
        # self.sim.physics_material = self.scene.terrain.physics_material

        # settings for rsl env control
        self.episode_length_s = 20.0 # can be ignored
        self.is_finite_horizon = False
        self.actions.joint_pos.scale = 0.25

        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt

def camera_follow(env):
    """调整 GUI 视角跟随第 0 台 Spot；将 Isaac 的 wxyz 四元数转为 SciPy 的 xyzw。"""
    # Remove conditional check and always use indexed naming
    robot_position = env.unwrapped.scene[f"spot"].data.root_state_w[0, :3].cpu().numpy()
    robot_orientation = env.unwrapped.scene[f"spot"].data.root_state_w[0, 3:7].cpu().numpy()
    rotation = R.from_quat([robot_orientation[1], robot_orientation[2], 
                            robot_orientation[3], robot_orientation[0]])
    yaw = rotation.as_euler('zyx')[0]
    yaw_rotation = R.from_euler('z', yaw).as_matrix()
    set_camera_view(
        yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position,
        robot_position
    )
