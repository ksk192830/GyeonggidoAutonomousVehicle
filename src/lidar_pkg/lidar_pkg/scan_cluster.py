import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy
)

from interfaces_pkg.msg import ParkingSpace
import numpy as np
import cv2
import math

# Biggest cluster  -> red (car 1)
# Second cluster   -> blue (car 2)
# All others/noise -> green

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
ANGLE_OFFSET_RAD = math.pi / 2.0

# Parking space width threshold [m]
PARKING_WIDTH_M = 0.6
# =========================


class LidarVisualizerNode(Node):
    def __init__(self):
        super().__init__('scan_cluster')

        # DBSCAN parameters
        self.dbscan_eps = 0.5          # [m] neighborhood radius
        self.dbscan_min_samples = 3    # minimum points to form a cluster

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )

        # parameters
        self.declare_parameter('sub_topic', SUB_TOPIC_NAME)
        self.declare_parameter('window_name', WINDOW_NAME)
        self.declare_parameter('enable_viz', True)

        self.sub_topic = self.get_parameter('sub_topic').get_parameter_value().string_value
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.enable_viz = self.get_parameter('enable_viz').get_parameter_value().bool_value

        # ParkingSpace publisher
        self.pub = self.create_publisher(ParkingSpace, "/parking_space", 10)

        # subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            self.sub_topic,
            self.scan_callback,
            qos_profile
        )

        self.get_logger().info(
            f'LidarVisualizerNode: subscribing to {self.sub_topic}, enable_viz={self.enable_viz}'
        )

        # canvas config
        self.img_size = IMAGE_SIZE
        self.center = (self.img_size // 2, self.img_size // 2)
        self.margin = 20
        self.scale = (self.img_size // 2 - self.margin) / DISPLAY_RANGE_MAX

        # angle guides (deg, label)
        self.angle_guides = [
            (0.0, "0"),
            (120.0, "120"),
            (240.0, "240"),
        ]

        # car image
        self.car_img = self.load_car_image(CAR_IMAGE_PATH)

        # OpenCV window only if visualization enabled
        if self.enable_viz:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, IMAGE_SIZE, IMAGE_SIZE)

    # ----------------- car image -----------------
    def load_car_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warn(f'Cannot load car image: {path}')
            return None

        img = cv2.resize(img, (CAR_IMAGE_WIDTH, CAR_IMAGE_HEIGHT))

        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.full_like(b, 255)
            img = cv2.merge((b, g, r, alpha))

        return img

    def overlay_car_center(self, background: np.ndarray) -> np.ndarray:
        if self.car_img is None:
            return background

        overlay = self.car_img
        h, w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]

        x1 = bg_w // 2 - w // 2
        y1 = bg_h // 2 - h
        x2 = x1 + w
        y2 = y1 + h

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
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        color = (120, 120, 120)

        ring_ratios = [0.25, 0.5, 0.75, 1.0]
        for ratio in ring_ratios:
            r_m = DISPLAY_RANGE_MAX * ratio
            radius_px = int(r_m * self.scale)

            cv2.circle(img, self.center, radius_px, (60, 60, 60), 1)

            label = f"{r_m:.1f} m"

            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_w, text_h = text_size

            text_x = self.center[0] + radius_px + 5
            text_y = self.center[1] - 5

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
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        max_r = DISPLAY_RANGE_MAX * self.scale
        label_r = max_r * 1.05

        for deg, label in self.angle_guides:
            angle_rad = math.radians(deg) + ANGLE_OFFSET_RAD

            x_end = int(self.center[0] + max_r * math.cos(angle_rad))
            y_end = int(self.center[1] - max_r * math.sin(angle_rad))

            line_color = (0, 0, 100)
            line_thickness = 2

            cv2.line(
                img,
                self.center,
                (x_end, y_end),
                line_color,
                line_thickness
            )

            x_text = int(self.center[0] + label_r * math.cos(angle_rad))
            y_text = int(self.center[1] - label_r * math.sin(angle_rad))

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

    # ----------------- DBSCAN implementation -----------------
    def dbscan(self, points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        n_points = points.shape[0]
        labels = np.full(n_points, -1, dtype=int)
        visited = np.zeros(n_points, dtype=bool)
        cluster_id = 0

        eps_sq = eps * eps

        for i in range(n_points):
            if visited[i]:
                continue
            visited[i] = True

            diffs = points - points[i]
            dist_sq = np.einsum('ij,ij->i', diffs, diffs)
            neighbors = np.where(dist_sq <= eps_sq)[0]

            if neighbors.size < min_samples:
                continue

            labels[i] = cluster_id
            seeds = list(neighbors)

            j = 0
            while j < len(seeds):
                p = seeds[j]
                if not visited[p]:
                    visited[p] = True
                    diffs_p = points - points[p]
                    dist_sq_p = np.einsum('ij,ij->i', diffs_p, diffs_p)
                    neighbors_p = np.where(dist_sq_p <= eps_sq)[0]

                    if neighbors_p.size >= min_samples:
                        for q in neighbors_p:
                            if q not in seeds:
                                seeds.append(q)

                if labels[p] == -1:
                    labels[p] = cluster_id
                j += 1

            cluster_id += 1

        return labels

    def cluster_points(self, points_xy):
        if not points_xy:
            return None, []

        pts = np.asarray(points_xy, dtype=np.float32)
        labels = self.dbscan(pts, self.dbscan_eps, self.dbscan_min_samples)

        valid = labels >= 0
        if not np.any(valid):
            return labels, []

        uniq, counts = np.unique(labels[valid], return_counts=True)
        order = np.argsort(-counts)
        main_clusters = uniq[order][:2].tolist()

        return labels, main_clusters

    # ----------------- LiDAR callback -----------------
    def scan_callback(self, msg: LaserScan):
        # 1) LiDAR → collect points
        points_xy = []
        points_pxpy = []

        angle = msg.angle_min
        for r in msg.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += msg.angle_increment
                continue

            if r < msg.range_min or r > DISPLAY_RANGE_MAX:
                angle += msg.angle_increment
                continue

            angle_visual = angle + ANGLE_OFFSET_RAD

            x = r * math.cos(angle_visual)
            y = r * math.sin(angle_visual)

            px = int(self.center[0] + x * self.scale)
            py = int(self.center[1] - y * self.scale)

            if 0 <= px < self.img_size and 0 <= py < self.img_size:
                points_xy.append((x, y))
                points_pxpy.append((px, py))

            angle += msg.angle_increment

        # 2) Clustering
        labels, main_clusters = self.cluster_points(points_xy) if points_xy else (None, [])

        # 3) Cluster centers (metric + pixel)
        cluster_centers_px = {}
        cluster_centers_xy = {}
        cluster_dist_m = {}

        if labels is not None and len(main_clusters) > 0:
            pts_xy = np.asarray(points_xy, dtype=np.float32)
            labels_np = np.asarray(labels, dtype=int)

            for cid in main_clusters:
                mask = labels_np == cid
                if not np.any(mask):
                    continue

                cluster_pts = pts_xy[mask]
                cx = float(np.mean(cluster_pts[:, 0]))
                cy = float(np.mean(cluster_pts[:, 1]))
                dist = math.hypot(cx, cy)

                px_center = int(self.center[0] + cx * self.scale)
                py_center = int(self.center[1] - cy * self.scale)

                cluster_centers_px[cid] = (px_center, py_center)
                cluster_centers_xy[cid] = (cx, cy)
                cluster_dist_m[cid] = dist

        # 4) Parking space detection (metric 기준)
        parking_found = False
        parking_x = 0.0
        parking_y = 0.0
        parking_yaw = 0.0

        if len(main_clusters) >= 2:
            cid1, cid2 = main_clusters[:2]

            if cid1 in cluster_centers_xy and cid2 in cluster_centers_xy:
                x1, y1 = cluster_centers_xy[cid1]
                x2, y2 = cluster_centers_xy[cid2]

                gap_m = math.hypot(x2 - x1, y2 - y1)

                if gap_m >= PARKING_WIDTH_M:
                    parking_found = True
                    parking_x = 0.5 * (x1 + x2)
                    parking_y = 0.5 * (y1 + y2)
                    # yaw: 차 두 대를 잇는 방향 (rad)
                    parking_yaw = math.atan2(y2 - y1, x2 - x1)


                    # ---- 라디안 → degree 변환 ----
                    yaw_deg = math.degrees(parking_yaw)

                    # ---- 범위 0 ≤ yaw_deg < 180 변환 ----
                    # atan2 결과는 -180~180 이므로 음수면 +180
                    if yaw_deg < 0:
                        yaw_deg += 180.0

                    # 180도 넘어가면 180도 경계 안으로 조정
                    yaw_deg = yaw_deg % 180.0

                    # ---- 발행은 degree 단위로 ----
                    parking_yaw = yaw_deg
                                        

        # 5) Always publish ParkingSpace (found or not)
        self.publish_space(parking_found, parking_x, parking_y, parking_yaw)

        # 6) If visualization disabled, stop here
        if not self.enable_viz:
            return

        # 7) Visualization
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        self.draw_range_rings_with_labels(img)

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

        self.draw_angle_guides(img)

        color_cluster_1 = (0, 0, 255)
        color_cluster_2 = (255, 0, 0)
        color_noise = (0, 255, 0)

        # Draw points
        if labels is None or len(main_clusters) == 0:
            for (px, py) in points_pxpy:
                cv2.circle(img, (px, py), 2, color_noise, -1)
        else:
            for idx, (px, py) in enumerate(points_pxpy):
                cid = labels[idx]
                if cid == -1:
                    color = color_noise
                elif cid == main_clusters[0]:
                    color = color_cluster_1
                elif len(main_clusters) > 1 and cid == main_clusters[1]:
                    color = color_cluster_2
                else:
                    color = color_noise

                cv2.circle(img, (px, py), 2, color, -1)

        # Draw main clusters (car1, car2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for i, cid in enumerate(main_clusters[:2]):
            if cid not in cluster_centers_px:
                continue

            px_center, py_center = cluster_centers_px[cid]
            dist_m = cluster_dist_m.get(cid, 0.0)

            if i == 0:
                color = color_cluster_1
                label = f"car1 {dist_m:.1f}m"
                offset = (-50, -10)
            else:
                color = color_cluster_2
                label = f"car2 {dist_m:.1f}m"
                offset = (10, 10)

            cv2.circle(img, (px_center, py_center), 8, color, -1)
            cv2.circle(img, (px_center, py_center), 25, color, 2)

            tx = max(5, min(self.img_size - 5, px_center + offset[0]))
            ty = max(15, min(self.img_size - 5, py_center + offset[1]))

            cv2.putText(
                img,
                label,
                (tx, ty),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        # === PARKING SPACE VISUALIZATION ===
        if parking_found and len(main_clusters) >= 2:
            cid1, cid2 = main_clusters[:2]
            if cid1 in cluster_centers_px and cid2 in cluster_centers_px:
                px1, py1 = cluster_centers_px[cid1]
                px2, py2 = cluster_centers_px[cid2]

                vx = px2 - px1
                vy = py2 - py1
                norm = math.hypot(vx, vy)

                if norm > 1e-6:
                    ux = vx / norm
                    uy = vy / norm

                    half_len_pix = (PARKING_WIDTH_M * self.scale) / 2.0

                    mx = (px1 + px2) / 2.0
                    my = (py1 + py2) / 2.0

                    p1 = (
                        int(mx - ux * half_len_pix),
                        int(my - uy * half_len_pix)
                    )
                    p2 = (
                        int(mx + ux * half_len_pix),
                        int(my + uy * half_len_pix)
                    )

                    # Cyan line for parking space (BGR: (255, 255, 0))
                    cv2.line(img, p1, p2, (255, 255, 0), 3)

        # ego vehicle center
        cv2.circle(img, self.center, 4, (0, 0, 255), -1)

        img = self.overlay_car_center(img)

        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1)
        if key == 27:
            self.get_logger().info('ESC pressed, shutting down visualizer')
            rclpy.shutdown()

    def publish_space(self, found, x=0.0, y=0.0, yaw=0.0):
        msg = ParkingSpace()
        msg.found = found
        msg.x = x
        msg.y = y
        msg.yaw = yaw
        self.pub.publish(msg)

    def destroy_node(self):
        if self.enable_viz:
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
