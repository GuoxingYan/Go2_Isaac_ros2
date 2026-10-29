#  Isaac Sim Unitree Go2/Spot ROS2
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-orange.svg)](https://docs.ros.org/en/humble/index.html)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-red.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.1.1-purple.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Linux platform](https://img.shields.io/badge/platform-Ubuntu--22.04-green.svg)](https://releases.ubuntu.com/22.04/)

Welcome to Unitree Go2 Simulation repository for Isaac-Sim(4.5.0) and IsaacLab(2.1.0-1). This repository has been updated to make use of latest Nvidia libraries using [Migration Guide](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_orbit.html) and tested on multiple devices. 

<img src="media/intro.gif" width="1000"/> 


## Update: 
- Isaac-Sim 6.0.0 and IsaacLab 3.0.0-beta2 support coming very soon. Please note it will require Ubuntu 24.04 with ROS2 Jazzy in ```Python 3.12```.

---

## Table of Contents
1. [Installation Guide](#1-installation-guide)  
2. [Run Unitree Go2 Simulation](#2-run-unitree-go2/spot-simulation)  
3. [ROS2 Topics and Visualization](#3-ros2-topics-and-visualization)  
   - [Command and Control](#31-command-and-control)  
   - [Front Camera](#32-front-camera)  
   - [LIDAR](#33-lidar)  
   - [Odometry and Localization](#34-odometry-and-localization)  
4. [Simulation Environments & Settings](#4-simulation-environments--settings)
   - [Launch different robots](#41-launch-different-robots)
   - [Launch different simulation environments](#42-launch-different-simulation-environments)  
   - [Launch different Policy/Algorithms](#43-launch-different-policyalgorithms)  
   - [Load custom checkpoints](#44-load-custom-checkpoints)  
   - [Launch multiple robots in the environment](#45-launch-multiple-robots-in-the-environment)  
   - [Change Yolo settings](#46-change-yolo-settings)  
5. [MCP LLM and Web Portal Usage](#5-mcp-llm-and-web-portal-usage)  
6. [Acknowledgement](#6-acknowledgement)  
7. [Citation](#7-citation)  
8. [Contact](#8-contact)  

---

## 1. Installation Guide
**Step I:** Please follow the [Isaac Lab official documentation](https://isaac-sim.github.io/IsaacLab//v2.1.1/source/setup/installation/index.html) to install the Isaac Sim 4.5.0 and Isaac Lab 2.1.0 or 2.1.1 .

**Step II:** Please install [ROS2 Humble](https://docs.ros.org/en/humble/index.html) with the official installation guide.

**Step III:** Install the prerequisite C extension in the conda environment. 
```
# Assuming you are using default conda env name from IsaacLab (env_isaaclab)
conda activate env_isaaclab     
conda install -c conda-forge libstdcxx-ng
```

**Step IV:** Clone this repo to your local directory.
```
git clone https://github.com/sallu-786/Go2_Isaac_ros2/
```

**Step V:** Install Dependencies for computer vision tasks
```
conda activate env_isaaclab  
pip install -r requirements.txt
```

## 2. Run Unitree Go2/Spot Simulation 
To run the simulation, please use the following command:
```
cd ~/Go2_Isaac_ros2
python main.py
```
Once the simulation is loaded, the robot can be teleoperated by the keyboard:

```W```: Forward, ```A```: Left, ```S```: Backward, ```D```: Right, ```Z```: Left Turn, ```C```: Right Turn.

To set your own custom keys make changes in [go2_ctrl.py](<quadruped/go2/go2_ctrl.py>)/[spot_ctrl.py](<quadruped/spot/spot_ctrl.py>) 


![preview](<media/preview.png>)

## 3. ROS2 Topics and Visualization
After launching the simulation, Open a new terminal and visualize the ROS2 data in ```Rviz2```:
```
cd ~/Go2_Isaac_ros2/rviz/
rviz2 -d go2.rviz         #use spot.rviz for spot
```

you should see something like this


![rviz](<media/rviz.png>)


If you dont see anything click add button and from topic list select topic you want to visualize. If you have more than 1 robot in environment you manually have to add topics in rviz.


![rviz add](<media/rviz_add.png>)

### 3.5 本项目的实际启动与验证命令

请使用 `rviz2`，不是 `viz2`。先在一个终端启动仿真，再在另一个终端启动 RViz。

当前 `cfg/sim.yaml` 的默认配置是 `robot: go2` 时，启动 Go2：

```bash
cd /data/projects/Sim2Real/1029_4.5/Go2_Isaac_ros2
source /data/anaconda3/etc/profile.d/conda.sh
conda activate go2_codex
source /data/apps/ros2-humble/activate_sim.sh
OMNI_KIT_ACCEPT_EULA=yes python main.py
```

另开终端查看 Go2：

```bash
cd /data/projects/Sim2Real/1029_4.5/Go2_Isaac_ros2
source /data/anaconda3/etc/profile.d/conda.sh
conda activate go2_codex
source /data/apps/ros2-humble/activate_sim.sh
rviz2 -d rviz/go2.rviz
```

如果出现：

```text
rviz2: symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0: undefined symbol: __libc_pthread_init
```

说明当前终端继承了 Snap 的库路径或预加载库。用干净的环境重新加载 ROS 2：

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH bash -lc '
  source /data/anaconda3/etc/profile.d/conda.sh
  conda activate go2_codex
  source /data/apps/ros2-humble/activate_sim.sh
  rviz2 -d /data/projects/Sim2Real/1029_4.5/Go2_Isaac_ros2/rviz/go2.rviz
'
```

Spot 的命令只需把最后一行替换为：

```bash
rviz2 -d /data/projects/Sim2Real/1029_4.5/Go2_Isaac_ros2/rviz/spot.rviz
```

Go2 的 ROS 2 话题前缀是 `/unitree_go2_0/`。可以用下面的命令确认消息正在发布：

```bash
ros2 topic list | grep unitree_go2_0
ros2 topic hz /unitree_go2_0/front_cam/color_image
ros2 topic echo --once /unitree_go2_0/front_cam/color_image --field height
ros2 topic echo --once /unitree_go2_0/lidar/point_cloud --field width
```

预期相机图像高度为 `480`，点云宽度应大于 `0`。

切换到 Spot 时，先停止当前仿真，再修改 [cfg/sim.yaml](<cfg/sim.yaml>)：

```yaml
robot: spot
```

然后重新运行 `python main.py`，并使用：

```bash
rviz2 -d rviz/spot.rviz
```

Spot 的 ROS 2 话题前缀是 `/spot_0/`。例如：

```bash
ros2 topic hz /spot_0/front_cam/color_image
ros2 topic echo --once /spot_0/lidar/point_cloud --field width
```

`spot.rviz` 只订阅 `/spot_0/...`，`go2.rviz` 只订阅 `/unitree_go2_0/...`，两者不能交叉使用。当前入口在 `argparse` 阶段不会接受 `python main.py robot=go2` 这种命令行覆盖方式，请直接修改 `cfg/sim.yaml`。

如果 RViz 无法显示内容，请依次检查：

1. 仿真窗口是否仍在运行并播放。
2. RViz 的 `Fixed Frame` 是否为 `map`。
3. Display 中的 Topic 是否与当前机器人前缀一致。
4. `ros2 topic hz` 是否持续收到消息。

RViz 需要桌面图形会话；不要使用 `QT_QPA_PLATFORM=offscreen` 启动 RViz，否则可能出现 `Invalid parentWindowHandle`。`depth_cloud` 是旧配置项，当前桥接代码主要发布 `depth_image`，如果该 Display 报错可以在 RViz 中禁用。

Here is a categorized list of ROS 2 topics available for the Unitree Go2 and Spot.


**Note:** If using spot you will see spot_0 instead of unitree_go2_0. <**0**> indicates the index as there can be multiple robots:

### 3.1 Command and Control  
- `/unitree_go2_0/cmd_vel`:  Topic to send velocity commands to the robot for motion control.

### 3.2 Front Camera 
- `/unitree_go2_0/front_cam/color_image`: Publishes RGB color images
- `/unitree_go2_0/front_cam/depth_image`: Publishes depth images
- `/unitree_go2_0/front_cam/semantic_segmentation_image_vis`: Publishes semantic segmentation images 
- `/unitree_go2_0/front_cam/info`: Publishes camera information, including intrinsic parameters
- `unitree_go2_0/front_cam/detection_image`: Publishes images with Yolo-based object detection 

### 3.3 LIDAR  
- `/unitree_go2_0/lidar/point_cloud`:  Publishes a point cloud generated by the robot's LIDAR sensor.

### 3.4 Odometry and Localization  
- `/unitree_go2_0/odom`:  Publishes odometry data, including the robot's position, orientation, and velocity.
- `/unitree_go2_0/pose`:  Publishes the current pose of the robot in the world frame.


## 4. Simulation Environments & settings
The simulation environments and settings can be changed in [sim.yaml](<cfg/sim.yaml>) config file. 

### 4.1 Launch different robots:
 To change the robot, please change value of ```robot``` in  [sim.yaml](<cfg/sim.yaml>). Currently Spot and Go2 are available.
 At the moment you cannot select both at same time.

 Here is example of spawning 3 spot robots

  <img src="media/spot.png" width="1000"/>  

### 4.2 Launch different simulation environments
The current implementation contains a few environments which follows standard Isaac Sim method for importing USD environments(from cloud). To change the environment, please change the ```env_name``` in  [sim.yaml](<cfg/sim.yaml>). Current available environments:
- ```warehouse```: A simple warehouse environment in Isaac Sim. (for further options in warehouse category, change path value in function inside  [sim_env.py](<env/sim_env.py>) )
- ```obstacle```: An obstacle field environment.  (change value of variable ```num_obstacles``` inside [sim_env.py](<env/sim_env.py>) to make it dense or sparse)
- ```terrain```: A Terrain environment in Isaac Sim. (for further options in terrain category, change path value in ```create_terrain_env()``` function inside  [sim_env.py](<env/sim_env.py>))
- ```office```
- ```hospital```
- ```rivermark```

As of now there are 70 environments assets available in isaac-sim 4.5.0. see more at [Environment Assets](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/assets/usd_assets_environments.html)
  
### 4.3 Launch different Policy/Algorithms 
To Launch Policy other than default (```ActorCritic```), go to [go2_ctrl_cfg.py](<go2/go2_ctrl_cfg.py>)/[spot_ctrl_cfg.py](<quadruped/spot/spot_ctrl_cfg.py>) and use any of the following values 
  - ```ActorCriticRecurrent```
  - ```StudentTeacher```
  - ```StudentTeacherRecurrent```

Algorithm by default is ```PPO``` you may also use ```Distillation```. For compatibility check and more details, please visit [API_docs](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)

### 4.4 Load custom checkpoints

Pretrained model (policy files) for flat and rough terrain are available in ```ckpts/unitree_go2``` folder. If you want to load your own policy, place the file in the folder and change the value for 'load_checkpoint' in [go2_ctrl_cfg.py](<go2/go2_ctrl_cfg.py>)/[spot_ctrl_cfg.py](<quadruped/spot/spot_ctrl_cfg.py>)..

### 4.5 Launch multiple robots in the environment
This repository supports running multiple Unitree Go2 robots and the number of robots can by changed by the ```num_envs``` parameter in the config file [sim.yaml](<cfg/sim.yaml>).

### 4.6 Change Yolo settings
Yolo models should be placed inside [yolo](<yolo>) folder Confidence threshold for classifying object is set to **0.7**. To customize model and confidence value, change the ```self.model``` and ```self.confidence_threshold``` 
inside [go2_ros2_bridge.py](<ros2/go2_ros2_bridge.py>)/[spot_ros2_bridge.py](<ros2/spot_ros2_bridge.py>).


## 5. MCP LLM and Web Portal Usage
Robot can also be controlled by either giving natural language commands from LLM or via web portal. not only that you can share it with others and they dont have to install anything. Read details in [README_MCP](<ros-mcp-server/README_MCP.md>)

![MCP](<media/mcp.gif>)

![Web_Portal](<media/portal.gif>)

## 6. Acknowledgement
Repository is built upon the work of [isaac-go2-ros2](https://github.com/Zhefan-Xu/isaac-go2-ros2)

Go2 controller is based on the RL controller implemented in [go2_omniverse](https://github.com/abizovnuralem/go2_omniverse).

MCP Control is based on [ros-mcp-server](https://github.com/lpigeon/ros-mcp-server)

work is supported by [Toyota-Boshoku](https://www.toyota-boshoku.com/)

## 7. Citation
```
@MISC{Suleman2025,
  author = "Muhammad Suleman",
  title = "Unitree Go2 in Isaac-Sim",
  year = "2025",
  url = "https://github.com/sallu-786/Go2_Isaac_ros2",
  note = "Version 1.2.1"
}
```
## 8. Contact
sulemanmuhammad08@gmail.com

