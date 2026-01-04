import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from ultralytics import YOLO
from interfaces_pkg.msg import Detection, DetectionArray, Mask, Point2D
from interfaces_pkg.msg import BoundingBox2D

import os


def color_for_id(idx: int) -> tuple:
    """
    Return a stable, visually distinct BGR color for a given integer id.
    Uses HSV hue wheel so different ids map to different colors.
    """
    hue = (idx * 37) % 180  # OpenCV HSV hue range: 0~179
    hsv = np.uint8([[[hue, 255, 255]]])  # full saturation/value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class YoloSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg')

        default_model_path = "/home/sg/contest_ws/src/camera_pkg/camera_pkg/model/best.pt"
        self.declare_parameter("model_path", default_model_path)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("threshold", 0.3)

        # Visualization options
        self.declare_parameter("mask_fill", True)         # True: fill + outline, False: outline only
        self.declare_parameter("mask_alpha", 0.25)        # transparency for filled masks
        self.declare_parameter("color_by", "class")       # "class" or "instance"

        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value

        self.mask_fill = self.get_parameter("mask_fill").get_parameter_value().bool_value
        self.mask_alpha = self.get_parameter("mask_alpha").get_parameter_value().double_value
        self.color_by = self.get_parameter("color_by").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.model = None

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.publisher = self.create_publisher(DetectionArray, "detections", qos)
        self.subscription = self.create_subscription(Image, "image_raw", self.image_cb, qos)
        self.vis_pub = self.create_publisher(Image, "seg_vis", qos)

        # Load model
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model not found: {self.model_path}")
            return

        try:
            self.model = YOLO(self.model_path)
            self.model.fuse()
            self.get_logger().info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return

    def image_cb(self, msg: Image):
        if self.model is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        try:
            results = self.model.predict(
                source=cv_image,
                device=self.device,
                conf=self.threshold,
                stream=False,
                verbose=False
            )[0].cpu()

            detections_msg = DetectionArray()
            detections_msg.header.stamp = self.get_clock().now().to_msg()
            detections_msg.header.frame_id = msg.header.frame_id

            # Build DetectionArray message
            for i in range(len(results.boxes)):
                det = Detection()
                det.class_id = int(results.boxes[i].cls)
                det.class_name = self.model.names[det.class_id]
                det.score = float(results.boxes[i].conf)

                bbox_xywh = results.boxes.xywh[i]
                det.bbox = BoundingBox2D()
                det.bbox.center.position.x = float(bbox_xywh[0])
                det.bbox.center.position.y = float(bbox_xywh[1])
                det.bbox.size.x = float(bbox_xywh[2])
                det.bbox.size.y = float(bbox_xywh[3])

                if results.masks is not None and i < len(results.masks.xy):
                    mask_msg = Mask()
                    mask_msg.width = results.orig_shape[1]
                    mask_msg.height = results.orig_shape[0]
                    for pt in results.masks.xy[i].tolist():
                        p = Point2D()
                        p.x = float(pt[0])
                        p.y = float(pt[1])
                        mask_msg.data.append(p)
                    det.mask = mask_msg

                detections_msg.detections.append(det)

            self.publisher.publish(detections_msg)

            # Visualization image
            vis = cv_image.copy()

            # Draw per-object overlays
            for i in range(len(results.boxes)):
                xyxy = results.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                class_id = int(results.boxes[i].cls)
                conf = float(results.boxes[i].conf)
                name = self.model.names[class_id]

                if self.color_by == "instance":
                    color = color_for_id(i)
                else:
                    color = color_for_id(class_id)

                # Draw bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # Label (with background)
                label = f"{name}:{conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_top = max(0, y1 - th - 8)
                cv2.rectangle(vis, (x1, y_top), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    vis, label, (x1 + 2, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                )

                # Draw mask (outline and optional fill)
                if results.masks is not None and i < len(results.masks.xy):
                    poly = results.masks.xy[i]
                    pts = poly.astype(int).reshape((-1, 1, 2))

                    if self.mask_fill:
                        overlay = vis.copy()
                        cv2.fillPoly(overlay, [pts], color)
                        alpha = float(self.mask_alpha)
                        vis = cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0)

                    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            vis_msg.header = msg.header
            self.vis_pub.publish(vis_msg)

        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
