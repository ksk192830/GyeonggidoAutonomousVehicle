#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

import cv2
import numpy as np


class ParkingFrontDetect(Node):
    def __init__(self):
        super().__init__("parking_front_detect")

        # Topics
        self.declare_parameter("image_topic", "/cam0/image_raw")
        self.declare_parameter("detection_topic", "/cam0/detections")
        self.declare_parameter("viz_topic", "/front_viz")

        # BEV ratios (normalized 0..1)
        self.declare_parameter(
            "persp_src",
            [0.25, 1.0, 0.75, 1.0, 0.62, 0.6, 0.38, 0.6],
        )
        self.declare_parameter(
            "persp_dst",
            [0.40, 1.0, 0.60, 1.0, 0.6, 0.8, 0.4, 0.8],
        )

        # Post-process
        self.declare_parameter("morph_kernel", 5)
        self.declare_parameter("slop", 0.3)

        image_topic = self.get_parameter("image_topic").value
        detection_topic = self.get_parameter("detection_topic").value
        viz_topic = self.get_parameter("viz_topic").value
        slop = float(self.get_parameter("slop").value)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.bridge = CvBridge()

        img_sub = Subscriber(self, Image, image_topic, qos_profile=qos)
        det_sub = Subscriber(self, DetectionArray, detection_topic, qos_profile=qos)

        self.ts = ApproximateTimeSynchronizer(
            [img_sub, det_sub], queue_size=10, slop=slop
        )
        self.ts.registerCallback(self.sync_callback)

        self.viz_pub = self.create_publisher(Image, viz_topic, qos)

        self.get_logger().info(
            f"parking_front_detect initialized. Sub: {image_topic}, {detection_topic} -> Pub: {viz_topic}"
        )

    def _get_perspective_matrix(self, w: int, h: int) -> np.ndarray:
        src = self.get_parameter("persp_src").value
        dst = self.get_parameter("persp_dst").value

        if len(src) != 8 or len(dst) != 8:
            raise ValueError(
                "persp_src and persp_dst must be length 8: [x0,y0,x1,y1,x2,y2,x3,y3]"
            )

        src_pts = np.float32([[src[i] * w, src[i + 1] * h] for i in range(0, 8, 2)])
        dst_pts = np.float32([[dst[i] * w, dst[i + 1] * h] for i in range(0, 8, 2)])
        return cv2.getPerspectiveTransform(src_pts, dst_pts)

    @staticmethod
    def _poly_to_mask(points, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), np.uint8)
        if points is None or len(points) < 3:
            return mask

        pts = np.array([[int(p.x), int(p.y)] for p in points], dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            return mask

        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _morph(self, m: np.ndarray) -> np.ndarray:
        k = int(self.get_parameter("morph_kernel").value)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        out = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kern)
        return out

    def sync_callback(self, img_msg: Image, det_msg: DetectionArray):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image convert failed: {e}")
            return

        h, w = frame.shape[:2]

        try:
            M = self._get_perspective_matrix(w, h)
        except Exception as e:
            self.get_logger().warn(f"Perspective transform invalid: {e}")
            return

        # Build per-class masks in original image space
        lot_mask = np.zeros((h, w), np.uint8)    # parking_lot
        space_mask = np.zeros((h, w), np.uint8)  # parking_space
        out_mask = np.zeros((h, w), np.uint8)    # out_line

        for det in det_msg.detections:
            cls = getattr(det, "class_name", "")
            if not hasattr(det, "mask") or det.mask is None:
                continue
            if not hasattr(det.mask, "data") or det.mask.data is None:
                continue

            m = self._poly_to_mask(det.mask.data, h, w)
            if m is None or m.size == 0:
                continue

            if cls == "parking_lot":
                lot_mask = cv2.bitwise_or(lot_mask, m)
            elif cls == "parking_space":
                space_mask = cv2.bitwise_or(space_mask, m)
            elif cls == "out_line":
                out_mask = cv2.bitwise_or(out_mask, m)

        # Warp each mask to BEV
        lot_bev = cv2.warpPerspective(lot_mask, M, (w, h), flags=cv2.INTER_NEAREST)
        space_bev = cv2.warpPerspective(space_mask, M, (w, h), flags=cv2.INTER_NEAREST)
        out_bev = cv2.warpPerspective(out_mask, M, (w, h), flags=cv2.INTER_NEAREST)

        # Morphology cleanup in BEV
        lot_bev = self._morph(lot_bev)
        space_bev = self._morph(space_bev)
        out_bev = self._morph(out_bev)

        # Compose visualization (BGR)
        bev = np.zeros((h, w, 3), np.uint8)  # background black

        # Order: parking_lot -> parking_space -> out_line
        lot_region = lot_bev > 0
        space_region = space_bev > 0
        out_region = out_bev > 0

        bev[lot_region] = (255, 255, 255)   # white
        bev[space_region] = (255, 0, 0)     # blue
        bev[out_region] = (0, 0, 255)       # red

        try:
            out_msg = self.bridge.cv2_to_imgmsg(bev, encoding="bgr8")
            out_msg.header = img_msg.header
            self.viz_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f"Publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ParkingFrontDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
