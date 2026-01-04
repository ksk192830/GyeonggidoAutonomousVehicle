import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy
)

import numpy as np
import cv2
import math

# ===== User settings =====
SUB_TOPIC_NAME = 'scan_raw'
WINDOW_NAME = 'Lidar Scan Viewer'

# Canvas size (pixels)
IMAGE_SIZE = 800

# Max display range [m] – 0 ~ DISPLAY_RANGE_MAX fills image radius
DISPLAY_RANGE_MAX = 4.0

# Car image path
CAR_IMAGE_PATH = '/home/kimmyungkyun/gyeonggi_ws/src/lidar_pkg/lidar_pkg/lib/car.png'

# Car image size
CAR_IMAGE_WIDTH = 120
CAR_IMAGE_HEIGHT = 120

# Global alpha for car image (0.0 ~ 1.0)
CAR_GLOBAL_ALPHA = 0.7

# Angle offset so that 0 rad points to "north" (up)
# 0 rad (scan) + 90 deg = up
ANGLE_OFFSET_RAD = math.pi / 2.0
# =========================


class LidarVisualizerNode(Node):
    def __init__(self):
        super().__init__('scan_viz')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )

        # parameters
        self.declare_parameter('sub_topic', SUB_TOPIC_NAME)
        self.declare_parameter('window_name', WINDOW_NAME)

        self.sub_topic = self.get_parameter('sub_topic').get_parameter_value().string_value
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value

        # subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            self.sub_topic,
            self.scan_callback,
            qos_profile
        )

        self.get_logger().info(f'LidarVisualizerNode: subscribing to {self.sub_topic}')

        # OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, IMAGE_SIZE, IMAGE_SIZE)

        # canvas config
        self.img_size = IMAGE_SIZE
        self.center = (self.img_size // 2, self.img_size // 2)
        self.margin = 20
        # map DISPLAY_RANGE_MAX [m] to radius [px]
        self.scale = (self.img_size // 2 - self.margin) / DISPLAY_RANGE_MAX

        # angle guides (deg, label)
        # 0° and 360° are the same direction, so they are combined
        self.angle_guides = [
            (0.0, "0"),
            (120.0, "120"),
            (240.0, "240"),
        ]

        # car image
        self.car_img = self.load_car_image(CAR_IMAGE_PATH)

    # ----------------- car image -----------------
    def load_car_image(self, path: str):
        """Load car image and normalize channels/size."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warn(f'Cannot load car image: {path}')
            return None

        # resize
        img = cv2.resize(img, (CAR_IMAGE_WIDTH, CAR_IMAGE_HEIGHT))

        # convert BGR to BGRA if no alpha channel
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.full_like(b, 255)
            img = cv2.merge((b, g, r, alpha))

        return img

    def overlay_car_center(self, background: np.ndarray) -> np.ndarray:
        """Overlay car image at the center (slightly above center)."""
        if self.car_img is None:
            return background

        overlay = self.car_img
        h, w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]

        x1 = bg_w // 2 - w // 2
        y1 = bg_h // 2 - h
        x2 = x1 + w
        y2 = y1 + h

        # range check
        if x1 < 0 or y1 < 0 or x2 > bg_w or y2 > bg_h:
            self.get_logger().warn('Car image exceeds canvas. Consider smaller size.')
            return background

        b, g, r, a = cv2.split(overlay)
        alpha = (a.astype(np.float32) / 255.0) * CAR_GLOBAL_ALPHA
        alpha = alpha[..., np.newaxis]

        roi = background[y1:y2, x1:x2].astype(np.float32)
        overlay_rgb = cv2.merge((b, g, r)).astype(np.float32)

        blended = roi * (1.0 - alpha) + overlay_rgb * alpha
        background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return background

    # ----------------- helper: draw rings + labels -----------------
    def draw_range_rings_with_labels(self, img: np.ndarray):
        """Draw distance rings and put distance labels in meters."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        color = (120, 120, 120)

        ring_ratios = [0.25, 0.5, 0.75, 1.0]
        for ratio in ring_ratios:
            r_m = DISPLAY_RANGE_MAX * ratio
            radius_px = int(r_m * self.scale)

            # draw ring
            cv2.circle(img, self.center, radius_px, (60, 60, 60), 1)

            # label text (e.g., "1.0 m")
            label = f"{r_m:.1f} m"

            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_w, text_h = text_size

            text_x = self.center[0] + radius_px + 5
            text_y = self.center[1] - 5

            # clamp inside image
            if text_x + text_w > self.img_size:
                text_x = self.img_size - text_w - 5
            if text_y - text_h < 0:
                text_y = text_h + 5

            cv2.putText(
                img,
                label,
                (text_x, text_y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

    # ----------------- helper: draw angle guides -----------------
    def draw_angle_guides(self, img: np.ndarray):
        """Draw radial lines and degree labels (0/360, 120, 240)."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        max_r = DISPLAY_RANGE_MAX * self.scale
        label_r = max_r * 1.05  # put label slightly outside the outer ring

        for deg, label in self.angle_guides:
            angle_rad = math.radians(deg) + ANGLE_OFFSET_RAD

            x_end = int(self.center[0] + max_r * math.cos(angle_rad))
            y_end = int(self.center[1] - max_r * math.sin(angle_rad))

            # highlight 0°/360° line a bit more
            if deg == 0.0:
                line_color = (0, 0, 100)
                line_thickness = 2
            else:
                line_color = (0, 0, 100)
                line_thickness = 2

            cv2.line(
                img,
                self.center,
                (x_end, y_end),
                line_color,
                line_thickness
            )

            # label position
            x_text = int(self.center[0] + label_r * math.cos(angle_rad))
            y_text = int(self.center[1] - label_r * math.sin(angle_rad))

            # clamp in image
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_w, text_h = text_size

            x_text = max(5, min(self.img_size - text_w - 5, x_text))
            y_text = max(text_h + 5, min(self.img_size - 5, y_text))

            cv2.putText(
                img,
                label,
                (x_text, y_text),
                font,
                font_scale,
                (100, 100, 100),
                thickness,
                cv2.LINE_AA
            )

    # ----------------- LiDAR callback -----------------
    def scan_callback(self, msg: LaserScan):
        """Visualize LaserScan as 2D image."""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # distance rings with labels
        self.draw_range_rings_with_labels(img)

        # guide axes (x, y)
        cv2.line(
            img,
            (0, self.center[1]),
            (self.img_size, self.center[1]),
            (70, 70, 70),
            1
        )
        cv2.line(
            img,
            (self.center[0], 0),
            (self.center[0], self.img_size),
            (70, 70, 70),
            1
        )

        # angle guides (0/360°, 120°, 240°) based on scan 0 rad + ANGLE_OFFSET_RAD
        self.draw_angle_guides(img)

        # draw lidar points
        angle = msg.angle_min
        for r in msg.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += msg.angle_increment
                continue

            if r < msg.range_min or r > DISPLAY_RANGE_MAX:
                angle += msg.angle_increment
                continue

            # rotate so that 0 rad points to "north" (up)
            angle_visual = angle + ANGLE_OFFSET_RAD

            # world coordinates: x (right/east), y (up/north)
            x = r * math.cos(angle_visual)
            y = r * math.sin(angle_visual)

            px = int(self.center[0] + x * self.scale)
            py = int(self.center[1] - y * self.scale)  # image y is inverted

            if 0 <= px < self.img_size and 0 <= py < self.img_size:
                cv2.circle(img, (px, py), 2, (0, 255, 0), -1)

            angle += msg.angle_increment

        # center point
        cv2.circle(img, self.center, 4, (0, 0, 255), -1)

        # overlay car image
        img = self.overlay_car_center(img)

        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            self.get_logger().info('ESC pressed, shutting down visualizer')
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
