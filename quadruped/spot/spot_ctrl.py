# Spot 的速度目标接口与 Go2 相同，但策略权重必须匹配 Spot 的观测和关节定义。
import os
import torch
import carb
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from quadruped.spot.spot_ctrl_cfg import spot_flat_cfg #, spot_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner
import torch

base_vel_cmd_input = None

# Initialize base_vel_cmd_input as a tensor when created
def init_base_vel_cmd(num_envs):
    """初始化每台机器人的 [vx, vy, wz] 机身速度目标，单位为米/秒、米/秒、弧度/秒。"""
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)

# Modify base_vel_cmd to use the tensor directly
def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    """将外部速度目标复制到环境设备，作为策略观测的一部分。"""
    global base_vel_cmd_input
    return base_vel_cmd_input.clone().to(env.device)

# Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
def sub_keyboard_event(event) -> bool:
    """用两组按键控制前两台机器人；任意键释放都会清空整个速度缓冲区。"""
    # 键盘与 ROS 写同一个张量，没有优先级仲裁；后处理的输入会覆盖先前目标。
    global base_vel_cmd_input

    lin_vel=1.0  # stable for 50 freq
    ang_vel=1.0  # stable for 50 freq
    
    if base_vel_cmd_input is not None:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:


#Default Keys----------------------------------------------------------------------------------------------------
            # Update tensor values for environment 0
            if event.input.name == 'W':
                base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'S':
                base_vel_cmd_input[0] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'A':
                base_vel_cmd_input[0] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
            elif event.input.name == 'D':
                base_vel_cmd_input[0] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
            elif event.input.name == 'Z':
                base_vel_cmd_input[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
            elif event.input.name == 'C':
                base_vel_cmd_input[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)

            # If there are multiple environments, handle inputs for env 1
            if base_vel_cmd_input.shape[0] > 1:
                if event.input.name == 'I':
                    base_vel_cmd_input[1] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'K':
                    base_vel_cmd_input[1] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'J':
                    base_vel_cmd_input[1] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'L':
                    base_vel_cmd_input[1] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'M':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
                elif event.input.name == '>':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
        
        # Reset commands to zero on key release
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            base_vel_cmd_input.zero_()
    return True

    
def get_rsl_flat_policy(cfg):
    """构建 Spot 平地环境并加载普通检查点，或在指定异常下回退到 TorchScript。"""

    cfg.observations.policy.height_scan = None
    # Spot 场景没有启用高度扫描器；平地策略也不需要这一观测项。
    env = gym.make("Isaac-Velocity-Flat-Spot-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = spot_flat_cfg
    ckpt_path = get_checkpoint_path(
        log_path=os.path.abspath("ckpts"),
        run_dir=agent_cfg["load_run"],
        checkpoint=agent_cfg["load_checkpoint"],
    )
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])

    try:
        # Standard PyTorch checkpoint
        ppo_runner.load(ckpt_path)
        policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
        print("Loaded standard PyTorch checkpoint.")

    except NotImplementedError:
        # 这里只捕获 NotImplementedError，并不是任何权重加载失败都会自动尝试 JIT。
        # TorchScript checkpoint
        policy = torch.jit.load(ckpt_path, map_location=agent_cfg["device"])
        policy.to(agent_cfg["device"])  # force model onto correct device
        print("Loaded JIT checkpoint.")

    return env, policy
