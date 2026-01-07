import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
import os
import re
import threading
import cv2

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver')

        self.declare_parameter('sub_topic', '/cam0/image_raw')
        self.declare_parameter('save_dir', '/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/lib/parking_rear_new')
        self.declare_parameter('frame_interval', 1)
        self.declare_parameter('image_format', 'png')
        self.declare_parameter("prefix", "image")
        self.declare_parameter("digits", 6)

        self.sub_topic = self.get_parameter('sub_topic').get_parameter_value().string_value
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.frame_interval = self.get_parameter('frame_interval').get_parameter_value().integer_value
        self.image_format = self.get_parameter("image_format").get_parameter_value().string_value.lower()
        self.prefix = self.get_parameter("prefix").get_parameter_value().string_value
        self.digits = int(self.get_parameter("digits").get_parameter_value().integer_value)

        if self.frame_interval <= 0:
            self.get_logger().warn("frame_interval must be >= 1. Forcing to 1.")
            self.frame_interval = 5

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.frame_count = 0

        self.index_lock = threading.Lock()
        self.next_index = self._load_next_index()

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
            f"saving every {self.frame_interval} frame(s) to '{self.save_dir}' as .{self.image_format}. "
            f"Next index: {self.next_index}"
        )

    def _load_next_index(self) -> int:
        """
        Scan save_dir for files like prefix + digits + .ext, e.g. image000123.png
        and set next_index to max+1.
        """
        pattern = re.compile(rf"^{re.escape(self.prefix)}(\d+)\.{re.escape(self.image_format)}$")
        max_idx = 0

        try:
            for name in os.listdir(self.save_dir):
                m = pattern.match(name)
                if not m:
                    continue
                try:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
        except Exception as e:
            self.get_logger().warn(f"Failed to scan directory '{self.save_dir}': {e}")
            max_idx = 0

        return max_idx + 1

    def image_callback(self, msg: Image):
        self.frame_count += 1

        if self.frame_count % self.frame_interval != 0:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return
        
        with self.index_lock:
            idx = self.next_index
            self.next_index += 1

        filename = f"{self.prefix}{idx:0{self.digits}d}.{self.image_format}"
        save_path = os.path.join(self.save_dir, filename)

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
