import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class ObstacleNode(Node):
    def __init__(self):
        super().__init__('obstacle')

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.image_sub = Subscriber(self, Image, '/cam0/image_raw', qos_profile=qos)
        self.detection_sub = Subscriber(self, DetectionArray, '/cam0/detections', qos_profile=qos)

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.detection_sub],
            queue_size=10,
            slop=0.5
        )
        self.ts.registerCallback(self.sync_callback)

        self.result_pub = self.create_publisher(String, 'obstacle_result', qos)
        self.mask_pub = self.create_publisher(Image, '/cam0/obstacle_mask', qos)

    def sync_callback(self, img_msg, det_msg):
        # ROS Image -> OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # 프레임 가로 정보
        width = cv_img.shape[1]
        x_min = int(width * (0.5 - 0.20))  # 중앙 50% 영역
        x_max = int(width * (0.5 + 0.20))

        # 화면 중앙에 가장 가까운 car 한 대 선택
        detected, area, cx_center, height_val = self.detect_obstacle(det_msg, width)

        if detected and cx_center is not None and x_min <= cx_center <= x_max:
            status = True
            final_area = area
        else:
            status = False
            final_area = 0
            height_val = 0

        status_msg = String(
            data=f"Detected: {status}, Area: {final_area}, Height: {height_val}"
        )
        self.result_pub.publish(status_msg)

        # 마스크 오버레이 (car 전부 그림)
        overlay = cv_img.copy()
        for det in det_msg.detections:
            if det.class_name != 'car':
                continue

            cx_box = int(det.bbox.center.position.x)
            cy_box = int(det.bbox.center.position.y)
            w = int(det.bbox.size.x)
            h = int(det.bbox.size.y)

            x1 = max(0, cx_box - w // 2)
            y1 = max(0, cy_box - h // 2)
            x2 = min(cv_img.shape[1], cx_box + w // 2)
            y2 = min(cv_img.shape[0], cy_box + h // 2)

            # 중앙 영역 안/밖에 따라 색만 다르게
            if x_min <= cx_box <= x_max:
                color = (0, 0, 255)  # red (inside)
            else:
                color = (255, 0, 0)  # blue (outside)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, cv_img, 1 - alpha, 0, cv_img)

        mask_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        self.mask_pub.publish(mask_msg)

    def detect_obstacle(self, detections: DetectionArray, img_width: int):
        """
        화면 가로 중앙(x = img_width/2)에 가장 가까운 car 하나를 선택.

        Returns:
            detected (bool)
            area (int)
            cx (int or None)
            height_val (int)  # bbox 중앙 하단 y좌표
        """
        center_x = img_width / 2.0

        best = None
        best_dist = None  # 중앙에서의 |cx - center_x|

        for det in detections.detections:
            if det.class_name != 'car':
                continue

            cx = int(det.bbox.center.position.x)
            cy = int(det.bbox.center.position.y)
            w = int(det.bbox.size.x)
            h = int(det.bbox.size.y)

            # bbox 중앙 하단 y좌표
            height_val = cy 

            dist_center = abs(cx - center_x)

            if best is None or dist_center < best_dist:
                best_dist = dist_center
                best = (cx, w, h, height_val)

        if best is None:
            return False, 0, None, 0

        cx, w, h, height_val = best
        area = w * h
        return True, area, cx, height_val


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
