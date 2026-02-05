#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import cv2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image

from interfaces_pkg.msg import DetectionArray, CrossWalk


class CrossWalkNode(Node):
    def __init__(self):
        super().__init__("cross_walk_node")

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        image_topic = "/cam0/image_raw"
        detection_topic = "/cam0/detections_cw"
        mask_topic = "/cam0/cross_walk_mask"
        result_topic = "cross_walk_result"

        self.image_sub = Subscriber(self, Image, image_topic, qos_profile=qos)
        self.detection_sub = Subscriber(self, DetectionArray, detection_topic, qos_profile=qos)
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.detection_sub],
            queue_size=10,
            slop=0.5,
        )
        self.ts.registerCallback(self.sync_callback)

        self.result_pub = self.create_publisher(CrossWalk, result_topic, qos)
        self.mask_pub = self.create_publisher(Image, mask_topic, qos)

    def sync_callback(self, img_msg: Image, det_msg: DetectionArray):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        found, height, boxes = self.extract_cross_walk(det_msg, cv_img.shape[1], cv_img.shape[0])

        result = CrossWalk()
        result.found = found
        result.height = int(height)
        self.result_pub.publish(result)

        overlay = cv_img.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)

        alpha = 0.5
        out_img = cv_img
        if boxes:
            cv2.addWeighted(overlay, alpha, cv_img, 1 - alpha, 0, out_img)

        mask_msg = self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8")
        self.mask_pub.publish(mask_msg)

    def extract_cross_walk(self, detections: DetectionArray, img_w: int, img_h: int):
        boxes = []
        top_y_min = img_h
        found = False

        for det in detections.detections:
            if det.class_name != "crosswalk":
                continue

            cx = int(det.bbox.center.position.x)
            cy = int(det.bbox.center.position.y)
            w = int(det.bbox.size.x)
            h = int(det.bbox.size.y)

            if w <= 0 or h <= 0:
                continue

            x1 = max(cx - w // 2, 0)
            y1 = max(cy - h // 2, 0)
            x2 = min(cx + w // 2, img_w)
            y2 = min(cy + h // 2, img_h)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append((x1, y1, x2, y2))
            found = True

            if y1 < top_y_min:
                top_y_min = y1

        if not found:
            top_y_min = 0

        return found, top_y_min, boxes



def main(args=None):
    rclpy.init(args=args)
    node = CrossWalkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
