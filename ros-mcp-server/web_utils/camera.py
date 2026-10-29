from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class RosImageSubscriber(Node,):
    """将 ROS 图像缓存为 OpenCV BGR 数组，供网页显示与模型描述使用。"""
    def __init__(self,topic_name:str):
        """创建两个图像订阅；当前二者订阅同一个传入话题，并不是不同图像源。"""
        super().__init__('image_subscriber')

        self.color_subscription = self.create_subscription(
            Image,                          
            f'{topic_name}',  
            self.color_image_callback,
            10
        )
        self.detection_subscription = self.create_subscription(
            Image,

            f'{topic_name}', 

            self.detection_image_callback,
            10
        )

        self.bridge = CvBridge()
        self.color_frame = None
        self.detection_frame = None

    def color_image_callback(self, msg):
        """更新彩色帧缓存，只保留最新图像。"""
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def detection_image_callback(self, msg):
        """更新检测帧缓存；实际是否包含检测框取决于订阅的话题。"""
        self.detection_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
