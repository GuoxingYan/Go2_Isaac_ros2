# Spot 桥接的数据流与 Go2 相同，使用 spot_i 话题前缀和 spot 场景资产键。
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
import quadruped.spot.spot_ctrl as spot_ctrl
#For YOLO detection
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, \
    BoundingBox2D, Pose2D
from visualization_msgs.msg import MarkerArray, Marker


ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
from isaacsim.ros2.bridge import collect_namespace, read_camera_info


class RobotDataManager(Node):
    """把 Spot 物理状态和渲染传感器结果接入 ROS 2，并处理速度控制和 YOLO 检测。"""
    def __init__(self, env, lidar_annotators, cameras, cfg):
        """初始化仿真时钟、各实例话题与 TF，并加载相机检测模型。"""
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
            prefix= f"spot_{i}"
            
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
                    lambda msg, idx=i: self.cmd_vel_callback(msg, idx), 10))
            #For yolo
            self.image_sub.append(self.create_subscription(Image, f"{prefix}/front_cam/color_image",
                lambda msg, idx=i: self.image_callback(msg, idx), 10))
            
            self.semantic_seg_img_sub.append(self.create_subscription(Image, 
                    f"{prefix}/front_cam/semantic_segmentation_image", 
                    lambda msg, idx=i: self.semantic_segmentation_callback(msg, idx), 10))
            
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
        """保留的小地图辅助入口；当前把发布器当消息读取，异常后返回原点。"""
    # Fetch robot pose from latest odometry (map frame)
        try:
            msg = self.pose_pub[env_idx]  # Or store latest PoseStamped in a variable
            x = msg.pose.position.x
            y = msg.pose.position.y
            return np.array([x, y])
        except:
            return np.array([0, 0])

    def world_to_map(self, pos):
        """按 map_scale 把米制平面位置映射到以图中心为原点的像素。"""
        # Convert world coordinates (meters) to map pixels
        map_x = int(self.map_size / 2 + pos[0]*self.map_scale)
        map_y = int(self.map_size / 2 - pos[1]*self.map_scale)
        return (map_x, map_y)

    def get_depth_at_pixel(self, env_idx, px, py):
        """返回固定 2 米占位值，目前不查询真实深度图。"""
        # Return depth value at given pixel from latest depth image
        # For simplicity, you can use a constant depth or subscribe to depth image
        return 2.0  # meters

    def transform_camera_to_map(self, robot_pos, depth, box):
        """以像素横向偏移和前向距离作简化定位，未使用机器人朝向和相机标定。"""
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
        """连接播放 tick、仿真时间读取和 ROS 时钟发布节点，生成 /clock。"""
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
        """异步执行 ROS 参数命令；返回值不代表子进程已成功设置 use_sim_time。"""
        # Define the command as a list
        command = ["ros2", "param", "set", "/robot_data_manager", "use_sim_time", "true"]

        # Run the command in a non-blocking way
        subprocess.Popen(command)  
        return True

    def create_static_transform(self):
        """发布机身到传感器的固定 TF；这些消息只描述坐标关系，不设置传感器安装位姿。"""
        # 相机 TF 平移为 (0.4, 0, 0.2)，但 spot_sensors.py 中实际相机平移是零。
        for i in range(self.num_envs):
            # LiDAR
            # Create and publish the transform
            lidar_broadcaster = StaticTransformBroadcaster(self)
            base_lidar_transform = TransformStamped()
            base_lidar_transform.header.stamp = self.get_clock().now().to_msg()
            base_lidar_transform.header.frame_id = f"spot_{i}/base_link"
            base_lidar_transform.child_frame_id = f"spot_{i}/lidar_frame"

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
            base_cam_transform.header.frame_id = f"spot_{i}/base_link"
            base_cam_transform.child_frame_id = f"spot_{i}/front_cam"

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
        """根据配置挂接图像和内参 writer；不会销毁主程序已经创建的相机。"""
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
        """发布真值里程计及 map→base_link；输入四元数 wxyz 重排为 ROS xyzw 字段。"""
        # 位姿来自仿真世界，速度来自机身局部系；该节点没有执行 SLAM 定位。
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = f"spot_{env_idx}/base_link"
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
        map_base_trans.child_frame_id = f"spot_{env_idx}/base_link"
        map_base_trans.transform.translation.x = base_pos[0].item()
        map_base_trans.transform.translation.y = base_pos[1].item()
        map_base_trans.transform.translation.z = base_pos[2].item()
        map_base_trans.transform.rotation.x = base_rot[1].item()
        map_base_trans.transform.rotation.y = base_rot[2].item()
        map_base_trans.transform.rotation.z = base_rot[3].item()
        map_base_trans.transform.rotation.w = base_rot[0].item()
        self.broadcaster.sendTransform(map_base_trans)
    

    def publish_pose(self, base_pos, base_rot, env_idx):
        """将仿真世界中的真值位姿发布为 map 坐标系 PoseStamped。"""
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
        """按 xyz 三个 float32 字段封装点云，并标记其雷达坐标系和时间戳。"""
        point_cloud = PointCloud2()
        point_cloud.header.frame_id = f"spot_{env_idx}/lidar_frame"
        point_cloud.header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud = point_cloud2.create_cloud(point_cloud.header, fields, points)
        self.lidar_pub[env_idx].publish(point_cloud)        

    def pub_ros2_data_callback(self):
        """一次发布所有 Spot 的位姿与速度；当前主循环使用带节流的另一入口。"""
        robot_data = self.env.unwrapped.scene["spot"].data
        for i in range(self.num_envs):
            self.publish_odom(robot_data.root_state_w[i, :3],
                              robot_data.root_state_w[i, 3:7],
                              robot_data.root_lin_vel_b[i],
                              robot_data.root_ang_vel_b[i],
                              i)
            self.publish_pose(robot_data.root_state_w[i, :3],
                              robot_data.root_state_w[i, 3:7], i)

    def pub_lidar_data_callback(self):
        """直接读取各实例雷达扫描缓冲并发布；当前没有绑定定时器。"""
        for i in range(self.num_envs):
            self.publish_lidar_data(self.lidar_annotators[i].get_data()["data"].reshape(-1, 3), i)

    def pub_ros2_data(self):
        """按墙上时间检查状态和点云的发布周期，实际频率受主循环调用频率限制。"""
        # 目标频率为状态 50 Hz、雷达 15 Hz；消息时间戳另由 ROS 仿真时钟提供。
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
            robot_data = self.env.unwrapped.scene["spot"].data
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
        """将机身 vx、vy、wz 速度目标写入共享张量；当前没有超时停止和输入仲裁。"""
        spot_ctrl.base_vel_cmd_input[env_idx][0] = msg.linear.x
        spot_ctrl.base_vel_cmd_input[env_idx][1] = msg.linear.y
        spot_ctrl.base_vel_cmd_input[env_idx][2] = msg.angular.z
    
    def semantic_segmentation_callback(self, img, env_idx):
        """把语义 ID 图转换为伪彩色图供可视化，颜色不是固定类别表。"""
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
        """保留的相机 OmniGraph 方案，默认未调用；其中路径需与实际 body 安装路径核对。"""
        for i in range(self.num_envs):

            color_topic_name = f"spot_{i}/front_cam/color_image"
            depth_topic_name = f"spot_{i}/front_cam/depth_image"
            frame_id = f"spot_{i}/front_cam"
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
                        ("IsaacCreateRenderProduct.inputs:cameraPrim", f"/World/envs/env_{i}/spot/base/front_cam"),
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
        """挂接 RGB 渲染 writer，让渲染管线持续发布彩色图像。"""
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"spot_{i}/front_cam/color_image"
            frame_id = f"spot_{i}/front_cam"

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
        """发布到成像平面的深度图；图像数据来自渲染而非机身状态张量。"""
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"spot_{i}/front_cam/depth_image"
            frame_id = f"spot_{i}/front_cam"


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
        """发布场景语义像素图及 ID 到标签的映射，依赖 USD 上的语义标注。"""
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"spot_{i}/front_cam/semantic_segmentation_image"
            label_topic_name = f"spot_{i}/front_cam/semantic_segmentation_label"                       
            frame_id = f"spot_{i}/front_cam"


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
        """挂接深度相机点云 writer；默认发布配置未调用本方法。"""
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"spot_{i}/front_cam/depth_cloud"
            frame_id = f"spot_{i}/front_cam"

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
        """发布相机尺寸与 K/R/P 标定矩阵；当前 frame_id 为不带前缀的 front_cam。"""
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"spot_{i}/front_cam/info"
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
        # 同步 YOLO 会阻塞处理此回调的主线程；当前置信度阈值是 0.5。
        # Spot 的 class_id 发布数字字符串，Go2 发布类别名称，两者当前表示不同。
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
                    hypothesis.hypothesis.class_id = str(class_id)
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
                    marker.header.frame_id = f"spot_{env_idx}/front_cam"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.id = idx
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    # Position marker in 3D space (simplified projection)
                    marker.pose.position.x = 2.0  # 2 meters in front of camera
                    # 固定距离的 Marker 仅是显示示意，不能用于真实三维测距或避障。
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
