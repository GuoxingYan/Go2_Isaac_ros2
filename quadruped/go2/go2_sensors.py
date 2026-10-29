import omni
import numpy as np
from pxr import Gf
import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

class SensorManager:
    """为每个 Go2 实例安装渲染传感器；ROS 发布由 RobotDataManager 单独负责。"""
    def __init__(self, num_envs):
        """记录机器人数量，使返回列表的索引与环境索引一致。"""
        self.num_envs = num_envs

    def add_rtx_lidar(self):
        """在机身上安装 VLS128 配置的 RTX 雷达，返回用于读取扫描结果的标注器列表。"""
        # RTX 雷达依赖渲染管线进行射线计算；不是从机器人状态张量直接生成点云。
        lidar_annotators = []
        for env_idx in range(self.num_envs):
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path=f"/unitree_go2_{env_idx}/lidar",  
                parent=f"/World/envs/env_{env_idx}/Go2/base",
                config="Velodyne_VLS128",
                translation=(0.2, 0, 0.2),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
            )

            annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
            # render product 是渲染输出管线；此处 1×1 只是挂接载体，不是点云分辨率。
            hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
            annotator.attach(hydra_texture.path)
            lidar_annotators.append(annotator)
        return lidar_annotators

    def add_camera(self, freq):
        """在各机身前方创建 640×480 相机；安装平移相对于父 prim，单位为米。"""
        # frequency 是请求的采样频率，实际新帧仍取决于主循环是否推进了渲染。
        cameras = []
        for env_idx in range(self.num_envs):
            camera = Camera(
                prim_path=f"/World/envs/env_{env_idx}/Go2/base/front_cam",
                translation=np.array([0.4, 0.0, 0.2]),
                frequency=freq,
                resolution=(640, 480),
                # resolution=(1280,720),
                orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
            )
            camera.initialize()
            camera.set_focal_length(1.5)
            cameras.append(camera)
        return cameras
