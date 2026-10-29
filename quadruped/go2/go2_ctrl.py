# 控制链：键盘/ROS → base_vel_cmd_input → 观测向量 → 策略 → 关节位置目标。
import os
import torch
import carb
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from quadruped.go2.go2_ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

base_vel_cmd_input = None

# Initialize base_vel_cmd_input as a tensor when created
def init_base_vel_cmd(num_envs):
    """创建形状为 (机器人数量, 3) 的速度缓冲区，列依次为 vx、vy、绕 z 轴角速度。"""
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)

# Modify base_vel_cmd to use the tensor directly
def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    """把机身坐标系目标速度复制到环境设备，供观测管理器拼接进策略输入。"""
    global base_vel_cmd_input
    return base_vel_cmd_input.clone().to(env.device)

# Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
def sub_keyboard_event(event) -> bool:
    """将按键转换为机身速度目标；平移单位为米/秒，转向单位为弧度/秒。

    第一组键控制索引 0，第二组键控制索引 1；任意按键释放都会清零所有机器人。
    每次按下覆盖整行，因此这里不是多个方向键叠加的控制方式。
    """
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
    """创建平地任务并加载 RSL-RL 权重，返回批量环境及可调用的推理策略。"""
    # 平地模型训练时没有高度扫描输入；保留它会改变网络输入维度和含义。
    cfg.observations.policy.height_scan = None
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)
    # 包装器适配 Isaac Lab 与 RSL-RL 的观测/动作接口；unwrapped 可访问底层场景。

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    # Runner 虽含 PPO 训练能力，此处只加载参数并取推理函数，没有启动训练。
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy

def get_rsl_rough_policy(cfg):
    """创建粗糙地形任务并加载对应权重；当前 main.py 默认不调用此入口。"""
    # 切换环境外观并不会调用本函数；高度扫描范围也必须与实际碰撞地形相符。
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy
