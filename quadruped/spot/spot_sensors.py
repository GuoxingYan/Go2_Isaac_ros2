import omni
import numpy as np
from pxr import Gf
import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

class SensorManager:
    """在 Spot 的 body prim 上安装相机和 RTX 雷达，返回与环境索引对应的列表。"""
    def __init__(self, num_envs):
        """记录需要创建传感器的机器人实例数。"""
        self.num_envs = num_envs

    def add_rtx_lidar(self):
        """创建基于 RTX 渲染的 VLS128 雷达，并挂接 CPU 扫描缓冲读取器。"""
        lidar_annotators = []
        for env_idx in range(self.num_envs):
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path=f"/spot_{env_idx}/lidar",  
                parent=f"/World/envs/env_{env_idx}/spot/body",
                config="Velodyne_VLS128",
                translation=(0.2, 0, 0.2),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
            )

            annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
            hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
            # 此 render product 驱动雷达计算，1×1 不表示雷达只有一个测距点。
            annotator.attach(hydra_texture.path)
            lidar_annotators.append(annotator)
        return lidar_annotators

    def add_camera(self, freq):
        """在 body 原点创建前置相机；分辨率是图像像素数，与窗口大小无关。"""
        # 当前安装平移为零，而桥接层相机 TF 使用 (0.4, 0, 0.2)，两处尚不一致。
        cameras = []
        for env_idx in range(self.num_envs):
            camera = Camera(
                prim_path=f"/World/envs/env_{env_idx}/spot/body/front_cam",
                translation=np.array([0.0, 0.0, 0.0]),
                frequency=freq,
                resolution=(640, 480),                           #Camera resolution
                #resolution=(1280,720),
                orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
            )
            camera.initialize()
            camera.set_focal_length(1.5)
            cameras.append(camera)
        return cameras
