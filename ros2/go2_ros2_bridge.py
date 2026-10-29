# 桥接层把仿真真值和渲染结果转换为 ROS 消息，并把 Twist 转回策略速度输入。
# ROS 话题、TF 坐标系名、USD prim 路径是三套命名体系，需要分别核对。
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import numpy as np
from cv_bridge import CvBridge
import cv2
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
import subprocess
import time
import quadruped.go2.go2_ctrl as go2_ctrl
#For YOLO detection
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, \
    BoundingBox2D, Pose2D
from visualization_msgs.msg import MarkerArray, Marker

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
from isaacsim.ros2.bridge import collect_namespace, read_camera_info


class RobotDataManager(Node):
    """管理所有 Go2 的话题、坐标变换、相机发布管线与同步 YOLO 回调。"""
    def __init__(self, env, lidar_annotators, cameras, cfg):
        """创建 /clock 图、每台机器人的发布订阅接口，并加载视觉检测模型。"""
        super().__init__("robot_data_manager")
        self.cfg = cfg
        self.create_ros_time_graph()
        sim_time_set = False
        while (rclpy.ok() and sim_time_set==False):
            sim_time_set = self.use_sim_time()

        self.env = env
        self.num_envs = env.unwrapped.scene.num_envs
        self.lidar_annotators = lidar_annotators
        self.cameras = cameras
        self.points = []
        self.step_size = 1
        self.node_namespace = ""         
        self.queue_size = 1
        self.confidence_threshold = 0.5  # Set your desired confidence threshold for YOLO detection
        
        # ROS2 Broadcaster
        self.broadcaster= TransformBroadcaster(self)
        
        # ROS2 Publisher
        self.odom_pub = []
        self.pose_pub = []
        self.lidar_pub = []
        self.semantic_seg_img_vis_pub = []
        self.det_pub = []
        self.vis_pub = []
        self.marker_pub = []

        # ROS2 Subscriber
        self.cmd_vel_sub = []
        self.color_img_sub = []
        self.depth_img_sub = []
        self.semantic_seg_img_sub = []
        self.image_sub = []

        # ROS2 Timer
        self.lidar_publish_timer = []

        #ROS2 SLAM
        # Mini-map parameters
        self.map_size = 500  # pixels
        self.map_scale = 50  # 1 meter = 50 pixels
        self.robot_radius = 5
        self.obj_radius = 3

        
        
        for i in range(self.num_envs):
            prefix= f"unitree_go2_{i}"
            
            self.odom_pub.append(self.create_publisher(Odometry, f"{prefix}/odom", 10))
            self.pose_pub.append(self.create_publisher(PoseStamped, f"{prefix}/pose", 10))
            self.lidar_pub.append(self.create_publisher(PointCloud2, 
                                                        f"{prefix}/lidar/point_cloud", 10))
            # Publishers for Yolo detections and visualization
            self.det_pub.append(self.create_publisher(Detection2DArray, 
                                    f"{prefix}/front_cam/detections",10))
            self.vis_pub.append(self.create_publisher(Image, 
                                    f"{prefix}/front_cam/detection_image", 10))
            self.marker_pub.append(self.create_publisher(MarkerArray,
                                    f"{prefix}/front_cam/detection_markers",10))
            # For semantic segmentation visualization
            self.semantic_seg_img_vis_pub.append(self.create_publisher(Image, 
                        f"{prefix}/front_cam/semantic_segmentation_image_vis", 10))
            
            self.cmd_vel_sub.append(self.create_subscription(Twist, f"{prefix}/cmd_vel", 
                    # 默认参数 idx=i 固定当前索引，避免 Python 闭包最终全部指向最后一台。
                    lambda msg, idx=i: self.cmd_vel_callback(msg, idx), 10))
            #For yolo
            self.image_sub.append(self.create_subscription(Image, f"{prefix}/front_cam/color_image",
                lambda msg, idx=i: self.image_callback(msg, idx), 10))
            
            self.semantic_seg_img_sub.append(self.create_subscription(Image, 
                    f"{prefix}/front_cam/semantic_segmentation_image", 
                    lambda msg, idx=i: self.semantic_segmentation_callback(msg, idx), 10))
        
        # self.create_timer(0.1, self.pub_ros2_data_callback)
        # self.create_timer(0.1, self.pub_lidar_data_callback)

        # use wall time for lidar and odom pub
        self.odom_pose_freq = 50.0
        self.lidar_freq = 15.0
        self.odom_pose_pub_time = time.time()
        self.lidar_pub_time = time.time() 
        self.create_static_transform()
        self.create_camera_publisher()  

        # YOLO setup
        self.bridge = CvBridge()
        self.model = YOLO('ckpts/yolo/yolov8n.pt')  # model path




    def get_robot_position(self, env_idx):
        """保留的小地图辅助入口；当前读取的是发布器对象，异常时回退到原点。"""
        # pose_pub 存放 Publisher 而非 PoseStamped，不能把该回退值当作真实定位结果。
    # Fetch robot pose from latest odometry (map frame)
        try:
            msg = self.pose_pub[env_idx]  # Or store latest PoseStamped in a variable
            x = msg.pose.position.x
            y = msg.pose.position.y
            return np.array([x, y])
        except:
            return np.array([0, 0])

    def world_to_map(self, pos):
        """把米制平面位置转为小地图像素，图像 y 轴向下所以需要反号。"""
        # Convert world coordinates (meters) to map pixels
        map_x = int(self.map_size / 2 + pos[0]*self.map_scale)
        map_y = int(self.map_size / 2 - pos[1]*self.map_scale)
        return (map_x, map_y)

    def get_depth_at_pixel(self, env_idx, px, py):
        """返回占位深度 2 米；当前没有读取对应像素的深度相机结果。"""
        # Return depth value at given pixel from latest depth image
        # For simplicity, you can use a constant depth or subscribe to depth image
        return 2.0  # meters

    def transform_camera_to_map(self, robot_pos, depth, box):
        """用固定前向距离和像素偏移估算二维位置，仅供简化显示，不是标定后的反投影。"""
        # Convert camera detection to world coordinates (simplified)
        # Assumes objects are in front of robot
        # For real implementation, use TF2 + camera intrinsics + depth
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        # Simple projection forward
        obj_x = robot_pos[0] + depth
        obj_y = robot_pos[1] + (center_x - 320)/320.0  # scale horizontal
        return np.array([obj_x, obj_y])

 
    def create_ros_time_graph(self):
        """用 OmniGraph 将每次播放 tick 的仿真时间发布到 /clock。"""
        # OmniGraph 的 CONNECT 连数据/执行端口，不是 Python 函数调用关系。
        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    # ("OnPlayBack", "omni.graph.action.OnImpulseEvent"),
                ],
                og.Controller.Keys.CONNECT: [
                    # Connecting execution of OnImpulseEvent node to PublishClock so it will only publish when an impulse event is triggered
                    # ("OnPlayBack.outputs:tick", "PublishClock.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),

                    # Connecting simulationTime data of ReadSimTime to the clock publisher node
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Assigning topic name to clock publisher
                    ("PublishClock.inputs:topicName", "/clock"),
                ],
            },
        )

    def use_sim_time(self):
        """异步启动 ROS CLI 设置仿真时间；返回 True 只表示已启动命令，没有确认设置成功。"""
        # Define the command as a list
        command = ["ros2", "param", "set", "/robot_data_manager", "use_sim_time", "true"]

        # Run the command in a non-blocking way
        subprocess.Popen(command)  
        return True

    def create_static_transform(self):
        """广播 base_link 到雷达及相机的固定安装关系，平移为米，旋转为 xyzw 四元数。"""
        # TF 告诉下游数据在哪个坐标系中；它不会移动 USD 里的真实传感器。
        for i in range(self.num_envs):
            # LiDAR
            # Create and publish the transform
            lidar_broadcaster = StaticTransformBroadcaster(self)
            base_lidar_transform = TransformStamped()
            base_lidar_transform.header.stamp = self.get_clock().now().to_msg()
            base_lidar_transform.header.frame_id = f"unitree_go2_{i}/base_link"
            base_lidar_transform.child_frame_id = f"unitree_go2_{i}/lidar_frame"

            # Translation
            base_lidar_transform.transform.translation.x = 0.2
            base_lidar_transform.transform.translation.y = 0.0
            base_lidar_transform.transform.translation.z = 0.2
            
            # Rotation 
            base_lidar_transform.transform.rotation.x = 0.0
            base_lidar_transform.transform.rotation.y = 0.0
            base_lidar_transform.transform.rotation.z = 0.0
            base_lidar_transform.transform.rotation.w = 1.0
            
            # Publish the transform
            lidar_broadcaster.sendTransform(base_lidar_transform)
    
            # -------------------------------------------------------------
            # Camera
            # Create and publish the transform
            camera_broadcaster = StaticTransformBroadcaster(self)
            base_cam_transform = TransformStamped()
            # base_cam_transform.header.stamp = self.get_clock().now().to_msg()
            base_cam_transform.header.frame_id = f"unitree_go2_{i}/base_link"
            base_cam_transform.child_frame_id = f"unitree_go2_{i}/front_cam"

            # Translation
            base_cam_transform.transform.translation.x = 0.4
            base_cam_transform.transform.translation.y = 0.0
            base_cam_transform.transform.translation.z = 0.2
            
            # Rotation 
            base_cam_transform.transform.rotation.x = -0.5
            base_cam_transform.transform.rotation.y = 0.5
            base_cam_transform.transform.rotation.z = -0.5
            base_cam_transform.transform.rotation.w = 0.5
            
            # Publish the transform
            camera_broadcaster.sendTransform(base_cam_transform)
    
    def create_camera_publisher(self):
        """按配置挂接 RGB、深度、语义发布器，并在启用相机时发布内参。"""
        # self.pub_image_graph()
        if (self.cfg.sensor.enable_camera):
            if (self.cfg.sensor.color_image):
                self.pub_color_image()
            if (self.cfg.sensor.depth_image):
                self.pub_depth_image()
            if (self.cfg.sensor.semantic_segmentation):
                self.pub_semantic_image()
            # self.pub_cam_depth_cloud()
            self.publish_camera_info()
    
    def publish_odom(self, base_pos, base_rot, base_lin_vel_b, base_ang_vel_b, env_idx):
        """发布仿真真值里程计和 map→base_link TF；位姿在世界系，速度在机身系。"""
        # Isaac 四元数顺序 wxyz，ROS 字段顺序 xyzw；这里显式重排，不能直接整体赋值。
        # 名字叫 odom，但位置直接取物理状态，并非轮速积分、视觉里程计或 SLAM 估计。
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = f"unitree_go2_{env_idx}/base_link"
        odom_msg.pose.pose.position.x = base_pos[0].item()
        odom_msg.pose.pose.position.y = base_pos[1].item()
        odom_msg.pose.pose.position.z = base_pos[2].item()
        odom_msg.pose.pose.orientation.x = base_rot[1].item()
        odom_msg.pose.pose.orientation.y = base_rot[2].item()
        odom_msg.pose.pose.orientation.z = base_rot[3].item()
        odom_msg.pose.pose.orientation.w = base_rot[0].item()
        odom_msg.twist.twist.linear.x = base_lin_vel_b[0].item()
        odom_msg.twist.twist.linear.y = base_lin_vel_b[1].item()
        odom_msg.twist.twist.linear.z = base_lin_vel_b[2].item()
        odom_msg.twist.twist.angular.x = base_ang_vel_b[0].item()
        odom_msg.twist.twist.angular.y = base_ang_vel_b[1].item()
        odom_msg.twist.twist.angular.z = base_ang_vel_b[2].item()
        self.odom_pub[env_idx].publish(odom_msg)

        # transform
        map_base_trans = TransformStamped()
        map_base_trans.header.stamp = self.get_clock().now().to_msg()
        map_base_trans.header.frame_id = "map"
        map_base_trans.child_frame_id = f"unitree_go2_{env_idx}/base_link"
        map_base_trans.transform.translation.x = base_pos[0].item()
        map_base_trans.transform.translation.y = base_pos[1].item()
        map_base_trans.transform.translation.z = base_pos[2].item()
        map_base_trans.transform.rotation.x = base_rot[1].item()
        map_base_trans.transform.rotation.y = base_rot[2].item()
        map_base_trans.transform.rotation.z = base_rot[3].item()
        map_base_trans.transform.rotation.w = base_rot[0].item()
        self.broadcaster.sendTransform(map_base_trans)


    # def publish_odom(self, base_pos, base_rot, base_lin_vel_b, base_ang_vel_b, env_idx):
    #     odom_msg = Odometry()
    #     odom_msg.header.stamp = self.get_clock().now().to_msg()
    #     # odom_msg.header.frame_id = "map"
    #     # odom_msg.child_frame_id = f"unitree_go2_{env_idx}/base_link"
    #     odom_msg.header.frame_id = "odom"            # parent
    #     odom_msg.child_frame_id = f"unitree_go2_{env_idx}/base_link"

    #     odom_msg.pose.pose.position.x = base_pos[0].item()
    #     odom_msg.pose.pose.position.y = base_pos[1].item()
    #     odom_msg.pose.pose.position.z = base_pos[2].item()
    #     odom_msg.pose.pose.orientation.x = base_rot[1].item()
    #     odom_msg.pose.pose.orientation.y = base_rot[2].item()
    #     odom_msg.pose.pose.orientation.z = base_rot[3].item()
    #     odom_msg.pose.pose.orientation.w = base_rot[0].item()
    #     odom_msg.twist.twist.linear.x = base_lin_vel_b[0].item()
    #     odom_msg.twist.twist.linear.y = base_lin_vel_b[1].item()
    #     odom_msg.twist.twist.linear.z = base_lin_vel_b[2].item()
    #     odom_msg.twist.twist.angular.x = base_ang_vel_b[0].item()
    #     odom_msg.twist.twist.angular.y = base_ang_vel_b[1].item()
    #     odom_msg.twist.twist.angular.z = base_ang_vel_b[2].item()
    #     self.odom_pub[env_idx].publish(odom_msg)

    #     # transform
    #     map_base_trans = TransformStamped()
    #     map_base_trans.header.stamp = self.get_clock().now().to_msg()
    #     map_base_trans.header.frame_id = "map"
    #     map_base_trans.child_frame_id = f"unitree_go2_{env_idx}/base_link"
    #     map_base_trans.transform.translation.x = base_pos[0].item()
    #     map_base_trans.transform.translation.y = base_pos[1].item()
    #     map_base_trans.transform.translation.z = base_pos[2].item()
    #     map_base_trans.transform.rotation.x = base_rot[1].item()
    #     map_base_trans.transform.rotation.y = base_rot[2].item()
    #     map_base_trans.transform.rotation.z = base_rot[3].item()
    #     map_base_trans.transform.rotation.w = base_rot[0].item()
    #     self.broadcaster.sendTransform(map_base_trans)
    
    def publish_pose(self, base_pos, base_rot, env_idx):
        """发布 map 坐标系下的机器人真值位置与姿态，不携带速度。"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = base_pos[0].item()
        pose_msg.pose.position.y = base_pos[1].item()
        pose_msg.pose.position.z = base_pos[2].item()
        pose_msg.pose.orientation.x = base_rot[1].item()
        pose_msg.pose.orientation.y = base_rot[2].item()
        pose_msg.pose.orientation.z = base_rot[3].item()
        pose_msg.pose.orientation.w = base_rot[0].item()
        self.pose_pub[env_idx].publish(pose_msg)

    def publish_lidar_data(self, points, env_idx):
        """把 N×3 点数组打包为 PointCloud2；每点 xyz 各占 4 字节，不附带强度字段。"""
        
        point_cloud = PointCloud2()
        point_cloud.header.frame_id = f"unitree_go2_{env_idx}/lidar_frame"
        point_cloud.header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud = point_cloud2.create_cloud(point_cloud.header, fields, points.tolist())
        
        self.lidar_pub[env_idx].publish(point_cloud)        

    def pub_ros2_data_callback(self):
        """批量发布当前机器人状态；当前主循环使用的是带节流的 pub_ros2_data。"""
        robot_data = self.env.unwrapped.scene["unitree_go2"].data
        for i in range(self.num_envs):
            self.publish_odom(robot_data.root_state_w[i, :3],
                              robot_data.root_state_w[i, 3:7],
                              robot_data.root_lin_vel_b[i],
                              robot_data.root_ang_vel_b[i],
                              i)
            self.publish_pose(robot_data.root_state_w[i, :3],
                              robot_data.root_state_w[i, 3:7], i)

    def pub_lidar_data_callback(self):
        """直接读取各雷达标注器并发布点云；当前未注册为定时器。"""
        for i in range(self.num_envs):
            self.publish_lidar_data(self.lidar_annotators[i].get_data()["data"].reshape(-1, 3), i)

    def pub_ros2_data(self):
        """按墙上时间节流状态和雷达发布；只能在主循环调用时检查是否到期。"""
        # 50 Hz/15 Hz 是目标上限：40 Hz 主循环不能实现 50 Hz 状态发布；超时不补发。
        # 节流用 time.time，消息时间戳用 ROS clock；两者在慢速仿真时并不等速。
        pub_odom_pose = False
        pub_lidar = False
        dt_odom_pose = time.time() - self.odom_pose_pub_time
        dt_lidar = time.time() - self.lidar_pub_time
        if (dt_odom_pose >= 1./self.odom_pose_freq):
            pub_odom_pose = True
        
        if (dt_lidar >= 1./self.lidar_freq):
            pub_lidar = True

        if (pub_odom_pose):
            self.odom_pose_pub_time = time.time()
            robot_data = self.env.unwrapped.scene["unitree_go2"].data
            for i in range(self.num_envs):
                self.publish_odom(robot_data.root_state_w[i, :3],
                                robot_data.root_state_w[i, 3:7],
                                robot_data.root_lin_vel_b[i],
                                robot_data.root_ang_vel_b[i],
                                i)
                self.publish_pose(robot_data.root_state_w[i, :3],
                                robot_data.root_state_w[i, 3:7], i)
        if (self.cfg.sensor.enable_lidar):
            if (pub_lidar):
                self.lidar_pub_time = time.time()
                for i in range(self.num_envs):
                    self.publish_lidar_data(self.lidar_annotators[i].get_data()["data"].reshape(-1, 3), i)

    def cmd_vel_callback(self, msg, env_idx):
        """将 Twist 的 vx、vy、wz 写入指定机器人的速度目标，供后续策略观测使用。"""
        # 当前没有命令超时自动清零；发送一次非零目标后，需要主动发送零速度停止。
        go2_ctrl.base_vel_cmd_input[env_idx][0] = msg.linear.x
        go2_ctrl.base_vel_cmd_input[env_idx][1] = msg.linear.y
        go2_ctrl.base_vel_cmd_input[env_idx][2] = msg.angular.z
    
    def semantic_segmentation_callback(self, img, env_idx):
        """将语义 ID 图归一化后着伪彩色便于显示；颜色并非稳定的类别编码。"""
        bridge = CvBridge()
        semantic_image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        semantic_image_normalized = (semantic_image / semantic_image.max() * 255).astype(np.uint8)

        # Apply a predefined colormap
        color_mapped_image = cv2.applyColorMap(semantic_image_normalized, cv2.COLORMAP_JET)
        # Convert to ROS Image
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(color_mapped_image, encoding='rgb8')
        self.semantic_seg_img_vis_pub[env_idx].publish(image_msg)


    def pub_image_graph(self):
        """保留的 OmniGraph 相机发布方案；当前默认采用下方 Replicator writer。"""
        for i in range(self.num_envs):

            color_topic_name = f"unitree_go2_{i}/front_cam/color_image"
            depth_topic_name = f"unitree_go2_{i}/front_cam/depth_image"
            # segmentation_topic_name = f"unitree_go2_{i}/front_cam/segmentation_image"
            # depth_cloud_topic_name = f"unitree_go2_{i}/front_cam/depth_cloud"
            frame_id = f"unitree_go2_{i}/front_cam"
            keys = og.Controller.Keys
            og.Controller.edit(
                {
                    "graph_path": "/CameraROS2Graph",
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("IsaacCreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("ROS2CameraHelperColor", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("ROS2CameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        # ("ROS2CameraHelperSegmentation", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        # ("ROS2CameraHelperDepthCloud", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],

                    keys.SET_VALUES: [
                        ("IsaacCreateRenderProduct.inputs:cameraPrim", f"/World/envs/env_{i}/Go2/base/front_cam"),
                        ("IsaacCreateRenderProduct.inputs:enabled", True),
                        ("IsaacCreateRenderProduct.inputs:height", 480),
                        ("IsaacCreateRenderProduct.inputs:width", 640),
                        
                        # color camera
                        ("ROS2CameraHelperColor.inputs:type", "rgb"),
                        ("ROS2CameraHelperColor.inputs:topicName", color_topic_name),
                        ("ROS2CameraHelperColor.inputs:frameId", frame_id),
                        ("ROS2CameraHelperColor.inputs:useSystemTime", False),

                        # depth camera
                        ("ROS2CameraHelperDepth.inputs:type", "depth"),
                        ("ROS2CameraHelperDepth.inputs:topicName", depth_topic_name),
                        ("ROS2CameraHelperDepth.inputs:frameId", frame_id),
                        ("ROS2CameraHelperDepth.inputs:useSystemTime", False),

                    ],

                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "IsaacCreateRenderProduct.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperColor.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperColor.inputs:renderProductPath"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperDepth.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperDepth.inputs:renderProductPath"),

                    ],

                },
            )

    def pub_color_image(self):
        """把相机渲染输出挂到 RGB ROS writer；后续帧由渲染管线自动发布。"""
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/color_image"
            frame_id = f"unitree_go2_{i}/front_cam"

            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
            
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name,
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)

    def pub_depth_image(self):
        """发布到成像平面的距离图；该深度不同于相机原点到表面的径向距离。"""
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/depth_image"
            frame_id = f"unitree_go2_{i}/front_cam"


            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                    sd.SensorType.DistanceToImagePlane.name
                                )
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)

    def pub_semantic_image(self):
        """分别挂接语义像素 ID 和标签映射发布器，类别来源于场景语义标注。"""
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/semantic_segmentation_image"
            label_topic_name = f"unitree_go2_{i}/front_cam/semantic_segmentation_label"                       
            frame_id = f"unitree_go2_{i}/front_cam"


            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                    sd.SensorType.SemanticSegmentation.name
                                )
            writer = rep.writers.get("ROS2PublishSemanticSegmentation")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name
            )
            writer.attach([render_product])

            semantic_writer = rep.writers.get(
               "SemanticSegmentationSD" + f"ROS2PublishSemanticLabels"
            )
            semantic_writer.initialize(
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=label_topic_name,
            )
            semantic_writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)

    def pub_cam_depth_cloud(self):
        """提供深度相机点云发布能力；当前 create_camera_publisher 没有启用此入口。"""
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/depth_cloud"
            frame_id = f"unitree_go2_{i}/front_cam"

            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                sd.SensorType.DistanceToImagePlane.name
            )

            writer = rep.writers.get(rv + "ROS2PublishPointCloud")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name,
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )

            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)   

    def publish_camera_info(self):
        """读取渲染相机内参并发布 CameraInfo，K/R/P 分别描述内参、校正及投影矩阵。"""
        # 此处 frame_id 仅为 front_cam；图像和 TF 使用带机器人前缀的名字，需要留意。
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/info"
            frame_id = self.cameras[i].prim_path.split("/")[-1] # This matches what the TF tree is publishing.

            writer = rep.writers.get("ROS2PublishCameraInfo")
            camera_info = read_camera_info(render_product_path=render_product)
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name,
                width=camera_info["width"],
                height=camera_info["height"],
                projectionType=camera_info["projectionType"],
                k=camera_info["k"].reshape([1, 9]),
                r=camera_info["r"].reshape([1, 9]),
                p=camera_info["p"].reshape([1, 12]),
                physicalDistortionModel=camera_info["physicalDistortionModel"],
                physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
            )
            writer.attach([render_product])

            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                "PostProcessDispatch" + "IsaacSimulationGate", render_product
            )

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)      

    def image_callback(self, msg, env_idx):
        """Process camera images and detect objects using YOLO."""
        # 在 spin_once 所在线程同步运行检测，耗时会直接增加主仿真循环时间。
        # 当前阈值为 self.confidence_threshold=0.5；主程序中的注释并未关闭这里的 YOLO。
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLO detection
            results = self.model(cv_image, verbose=False)
            
            # Create Detection2DArray message
            det_array_msg = Detection2DArray()
            det_array_msg.header = msg.header
            
            # Create marker array for 3D visualization
            marker_array = MarkerArray()
            
            # Process each detection
            for idx, result in enumerate(results[0].boxes.data):
                box = result[:4].cpu().numpy()  # Convert tensor to numpy array
                score = float(result[4])
                class_id = int(result[5])
                if score >= self.confidence_threshold: 
                    # Create 2D detection message
                    det = Detection2D()
                    det.header = msg.header
                    
                    # Set bounding box
                    det.bbox = BoundingBox2D()
                    center_pose = Pose2D()
                    center_pose.position.x = float((box[0] + box[2]) / 2)
                    center_pose.position.y = float((box[1] + box[3]) / 2)
                    center_pose.theta = 0.0
                    det.bbox.center = center_pose
                    det.bbox.size_x = float(box[2] - box[0])
                    det.bbox.size_y = float(box[3] - box[1])
                    
                    # Add class and score
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = self.model.names[class_id]#str(class_id)
                    hypothesis.hypothesis.score = score
                    det.results.append(hypothesis)
                    det_array_msg.detections.append(det)
                    
                    # Draw detection on image
                    cv2.rectangle(
                        cv_image, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Add label with class name and score
                    label = f"{self.model.names[class_id]}: {score:.2f}"
                    cv2.putText(
                        cv_image,
                        label,
                        (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    
                    # Create 3D marker for visualization
                    marker = Marker()
                    marker.header.frame_id = f"unitree_go2_{env_idx}/front_cam"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.id = idx
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    # Position marker in 3D space (simplified projection)
                    marker.pose.position.x = 2.0  # 2 meters in front of camera
                    # 三维 Marker 固定放在前方 2 米，未融合深度和内参，不是真实物体三维位置。
                    marker.pose.position.y = ((box[0] + box[2])/2 - 320) / 100.0  # scaled x offset from center
                    marker.pose.position.z = ((box[1] + box[3])/2 - 240) / 100.0  # scaled y offset from center
                    
                    # Set marker orientation
                    marker.pose.orientation.w = 1.0
                    
                    # Set marker size
                    marker.scale.x = 0.2
                    marker.scale.y = 0.2
                    marker.scale.z = 0.2
                    
                    # Set marker color
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.8
                    
                    marker_array.markers.append(marker)
            
            # Publish all results
            self.det_pub[env_idx].publish(det_array_msg)
            self.vis_pub[env_idx].publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            self.marker_pub[env_idx].publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')


    # def image_callback(self, msg, env_idx):
    #     try:
    #         # Convert ROS Image to OpenCV image
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
    #         # Run YOLO detection
    #         results = self.model(cv_image, verbose=False)
            
    #         # Mini-map (black background)
    #         map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
            
    #         # Get robot pose from odometry (map frame)
    #         robot_pos = self.get_robot_position(env_idx)  # implement this function
    #         robot_pixel = self.world_to_map(robot_pos)
    #         cv2.circle(map_img, robot_pixel, self.robot_radius, (0, 255, 0), -1)

    #         # Process each YOLO detection
    #         det_array_msg = Detection2DArray()
    #         det_array_msg.header = msg.header
    #         marker_array = MarkerArray()
    #         for idx, result in enumerate(results[0].boxes.data):
    #             box = result[:4].cpu().numpy()
    #             score = float(result[4])
    #             class_id = int(result[5])
    #             if score >= self.confidence_threshold:
    #                 # Draw detection on camera
    #                 cv2.rectangle(cv_image, (int(box[0]), int(box[1])), 
    #                             (int(box[2]), int(box[3])), (0, 255, 0), 2)
    #                 label = f"{self.model.names[class_id]}: {score:.2f}"
    #                 cv2.putText(cv_image, label, (int(box[0]), int(box[1]) - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #                 # Estimate object position in map frame using depth (simplified)
    #                 depth = self.get_depth_at_pixel(env_idx, int((box[0]+box[2])/2), 
    #                                                 int((box[1]+box[3])/2))
    #                 obj_world_pos = self.transform_camera_to_map(robot_pos, depth, box)
    #                 obj_pixel = self.world_to_map(obj_world_pos)
    #                 cv2.circle(map_img, obj_pixel, self.obj_radius, (0, 0, 255), -1)

    #         # Combine camera feed and mini-map
    #         combined = cv2.hconcat([cv_image, cv2.resize(map_img, (cv_image.shape[1], cv_image.shape[0]))])
    #         cv2.imshow(f"Env {env_idx} Camera + Mini-Map", combined)
    #         cv2.waitKey(1)

    #     except Exception as e:
    #         self.get_logger().error(f'Error in image_callback: {str(e)}')
