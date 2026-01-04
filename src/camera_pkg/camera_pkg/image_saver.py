import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
import os
from datetime import datetime


class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver')

        self.declare_parameter('sub_topic', '/cam0/image_raw')
        self.declare_parameter('save_dir', '/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/lib/mission5')
        self.declare_parameter('frame_interval', 1)
        self.declare_parameter('image_format', 'png')

        self.sub_topic = self.get_parameter('sub_topic').get_parameter_value().string_value
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.frame_interval = self.get_parameter('frame_interval').get_parameter_value().integer_value
        self.image_format = self.get_parameter('image_format').get_parameter_value().string_value

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.frame_count = 0

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )

        self.subscription = self.create_subscription(
            Image,
            self.sub_topic,
            self.image_callback,
            qos
        )

        self.get_logger().info(
            f"ImageSaverNode started. Subscribing to '{self.sub_topic}', "
            f"saving every {self.frame_interval} frame(s) to '{self.save_dir}' as .{self.image_format}"
        )

    def image_callback(self, msg: Image):
        self.frame_count += 1

        if self.frame_count % self.frame_interval != 0:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"frame_{self.frame_count:06d}_{timestamp}.{self.image_format}"
        save_path = os.path.join(self.save_dir, filename)

        import cv2
        success = cv2.imwrite(save_path, cv_image)

        if success:
            self.get_logger().info(f"Saved image: {save_path}")
        else:
            self.get_logger().warn(f"Failed to save image: {save_path}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
