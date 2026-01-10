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

        self.declare_parameter("image_topic", "/cam0/image_raw")
        self.declare_parameter("detection_topic", "/cam0/detections")
        self.declare_parameter("viz_topic", "/front_viz")

        self.declare_parameter(
            "persp_src",
            [0.25, 1.0, 0.75, 1.0, 0.62, 0.6, 0.38, 0.6],
        )
        self.declare_parameter(
            "persp_dst",
            [0.40, 1.0, 0.60, 1.0, 0.6, 0.8, 0.4, 0.8],
        )

        self.declare_parameter("morph_kernel", 7)
        self.declare_parameter("slop", 0.3)
        self.declare_parameter("fill_holes", True)

        self.declare_parameter("simplify_enable", True)
        self.declare_parameter("simplify_min_area_ratio", 0.002)
        self.declare_parameter("simplify_eps_ratio_space", 0.02)
        self.declare_parameter("simplify_eps_ratio_lot", 0.03)
        self.declare_parameter("simplify_max_vertices", 6)
        self.declare_parameter("simplify_extra_passes", 2)
        self.declare_parameter("draw_simplified_edges", True)

        self.declare_parameter("lot_components", 2)
        self.declare_parameter("lot_min_component_area_ratio", 0.001)

        self.declare_parameter("join_enable", True)
        self.declare_parameter("join_kernel", 13)
        self.declare_parameter("union_simplify_enable", True)
        self.declare_parameter("union_eps_ratio", 0.02)
        self.declare_parameter("union_max_vertices", 8)
        self.declare_parameter("union_extra_passes", 2)

        self.declare_parameter("space_force_convex", True)
        self.declare_parameter("convex_max_area_growth", 5)

        # === ADD: visualize BEV cut boundary (valid area) ===
        self.declare_parameter("draw_bev_cut_boundary", True)
        self.declare_parameter("bev_cut_boundary_thickness", 2)
        self.declare_parameter("bev_cut_boundary_color_bgr", [0, 255, 255])  # yellow
        self.declare_parameter("bev_cut_boundary_min_y_ratio", 0.5)  # ignore contours above this (like lane code)

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
            raise ValueError("persp_src/dst length must be 8")
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

    @staticmethod
    def _fill_holes(binary: np.ndarray) -> np.ndarray:
        b = (binary > 0).astype(np.uint8) * 255
        h, w = b.shape[:2]
        flood = b.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        return cv2.bitwise_or(b, cv2.bitwise_not(flood))

    def _postprocess_mask(self, m: np.ndarray) -> np.ndarray:
        out = (m > 0).astype(np.uint8) * 255
        out = self._morph(out)
        if bool(self.get_parameter("fill_holes").value):
            out = self._fill_holes(out)
        out = self._morph(out)
        return out

    @staticmethod
    def _keep_n_largest_components(binary: np.ndarray, n: int, min_area_px: int):
        b = (binary > 0).astype(np.uint8)
        if n <= 0:
            return []
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        if num_labels <= 1:
            return [(b * 255).astype(np.uint8)] if b.sum() > 0 else []

        areas = stats[1:, cv2.CC_STAT_AREA]
        valid = np.where(areas >= max(1, min_area_px))[0]
        if valid.size == 0:
            return []

        order = valid[np.argsort(areas[valid])[::-1]]
        order = order[: min(n, order.size)]

        comps = []
        for idx in order:
            label = idx + 1
            m = np.zeros_like(b, np.uint8)
            m[labels == label] = 255
            comps.append(m)
        return comps

    def _simplify_contour(self, cnt, eps_ratio: float, max_vertices: int, extra_passes: int):
        arc = cv2.arcLength(cnt, True)
        if arc <= 1.0:
            return None
        eps = max(1.0, eps_ratio * arc)
        approx = cv2.approxPolyDP(cnt, eps, True)
        for _ in range(int(max(0, extra_passes))):
            if approx is None or len(approx) <= max_vertices:
                break
            eps *= 1.5
            approx = cv2.approxPolyDP(cnt, eps, True)
        return approx

    def _simplify_to_polygon_mask(
        self,
        binary: np.ndarray,
        img_area: int,
        eps_ratio: float,
        max_vertices: int,
        extra_passes: int,
        force_convex: bool = False,
        convex_max_area_growth: float = 1.10,
    ):
        b = (binary > 0).astype(np.uint8) * 255
        if b.sum() == 0:
            return b, []

        min_area_ratio = float(self.get_parameter("simplify_min_area_ratio").value)
        min_area_px = int(max(1, img_area * max(0.0, min_area_ratio)))

        contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return b, []
        contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
        if not contours:
            return np.zeros_like(b, np.uint8), []

        cnt = max(contours, key=cv2.contourArea)
        orig_area = float(cv2.contourArea(cnt))

        if force_convex:
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            if orig_area > 1.0 and hull_area / orig_area <= max(1.0, convex_max_area_growth):
                cnt_use = hull
            else:
                cnt_use = cnt
        else:
            cnt_use = cnt

        approx = self._simplify_contour(cnt_use, eps_ratio, max_vertices, extra_passes)

        poly = []
        if approx is not None and len(approx) >= 3:
            poly = approx.reshape(-1, 2).astype(np.int32)

        out = np.zeros_like(b, np.uint8)
        if len(poly) >= 3:
            cv2.fillPoly(out, [poly], 255)
        return out, [poly] if len(poly) >= 3 else []

    @staticmethod
    def _draw_polys(img_bgr: np.ndarray, polys, color_bgr, thickness: int = 2):
        if not polys:
            return
        for poly in polys:
            if poly is None or len(poly) < 3:
                continue
            cv2.polylines(img_bgr, [poly], isClosed=True, color=color_bgr, thickness=thickness)

    @staticmethod
    def _kernel_rect(k: int):
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    def _join_masks(self, lot: np.ndarray, space: np.ndarray) -> np.ndarray:
        union = cv2.bitwise_or(lot, space)
        if not bool(self.get_parameter("join_enable").value):
            return union
        k = int(self.get_parameter("join_kernel").value)
        kern = self._kernel_rect(k)
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kern)
        if bool(self.get_parameter("fill_holes").value):
            union = self._fill_holes(union)
        union = cv2.morphologyEx(union, cv2.MORPH_OPEN, kern)
        return union

    # === ADD: compute valid-area mask (BEV) and draw its boundary ===
    def _draw_bev_cut_boundary(self, bev_bgr: np.ndarray, M: np.ndarray, w: int, h: int):
        if not bool(self.get_parameter("draw_bev_cut_boundary").value):
            return

        ones = np.ones((h, w), np.uint8) * 255
        valid = cv2.warpPerspective(ones, M, (w, h), flags=cv2.INTER_NEAREST) > 0

        inv = (~valid).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = self.get_parameter("bev_cut_boundary_color_bgr").value
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            color = [0, 255, 255]
        color = (int(color[0]), int(color[1]), int(color[2]))

        thickness = int(self.get_parameter("bev_cut_boundary_thickness").value)
        if thickness < 1:
            thickness = 1

        min_y_ratio = float(self.get_parameter("bev_cut_boundary_min_y_ratio").value)
        min_y = int(h * np.clip(min_y_ratio, 0.0, 1.0))

        for cnt in cnts:
            if cnt.size == 0:
                continue
            if int(cnt[:, :, 1].max()) < min_y:
                continue
            cv2.drawContours(bev_bgr, [cnt], -1, color, thickness)

        # optional outer border (same as lane code rectangle)
        cv2.rectangle(bev_bgr, (0, 0), (w - 1, h - 1), color, thickness)

    def sync_callback(self, img_msg: Image, det_msg: DetectionArray):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image convert failed: {e}")
            return

        h, w = frame.shape[:2]
        img_area = int(h * w)

        try:
            M = self._get_perspective_matrix(w, h)
        except Exception as e:
            self.get_logger().warn(f"Perspective transform invalid: {e}")
            return

        lot_mask = np.zeros((h, w), np.uint8)
        space_mask = np.zeros((h, w), np.uint8)
        out_mask = np.zeros((h, w), np.uint8)

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

        lot_bev = cv2.warpPerspective(lot_mask, M, (w, h), flags=cv2.INTER_NEAREST)
        space_bev = cv2.warpPerspective(space_mask, M, (w, h), flags=cv2.INTER_NEAREST)
        out_bev = cv2.warpPerspective(out_mask, M, (w, h), flags=cv2.INTER_NEAREST)

        lot_bev = self._postprocess_mask(lot_bev)
        space_bev = self._postprocess_mask(space_bev)
        out_bev = self._postprocess_mask(out_bev)

        lot_polys = []
        space_polys = []
        union_polys = []

        if bool(self.get_parameter("simplify_enable").value):
            eps_space = float(self.get_parameter("simplify_eps_ratio_space").value)
            eps_lot = float(self.get_parameter("simplify_eps_ratio_lot").value)
            max_v = int(self.get_parameter("simplify_max_vertices").value)
            extra = int(self.get_parameter("simplify_extra_passes").value)

            n = int(self.get_parameter("lot_components").value)
            min_comp_ratio = float(self.get_parameter("lot_min_component_area_ratio").value)
            min_comp_px = int(max(1, img_area * max(0.0, min_comp_ratio)))
            lot_components = self._keep_n_largest_components(lot_bev, n=n, min_area_px=min_comp_px)

            lot_merged = np.zeros_like(lot_bev, np.uint8)
            for comp in lot_components:
                comp_pp = self._postprocess_mask(comp)
                comp_simplified, polys = self._simplify_to_polygon_mask(
                    comp_pp, img_area=img_area, eps_ratio=eps_lot, max_vertices=max_v, extra_passes=extra
                )
                lot_merged = cv2.bitwise_or(lot_merged, comp_simplified)
                lot_polys.extend(polys)
            lot_bev = lot_merged

            force_convex = bool(self.get_parameter("space_force_convex").value)
            growth = float(self.get_parameter("convex_max_area_growth").value)

            space_bev, space_polys = self._simplify_to_polygon_mask(
                space_bev,
                img_area=img_area,
                eps_ratio=eps_space,
                max_vertices=max_v,
                extra_passes=extra,
                force_convex=force_convex,
                convex_max_area_growth=growth,
            )

        union_bev = self._join_masks(lot_bev, space_bev)

        if bool(self.get_parameter("union_simplify_enable").value):
            eps_u = float(self.get_parameter("union_eps_ratio").value)
            max_u = int(self.get_parameter("union_max_vertices").value)
            extra_u = int(self.get_parameter("union_extra_passes").value)
            union_bev, union_polys = self._simplify_to_polygon_mask(
                union_bev, img_area=img_area, eps_ratio=eps_u, max_vertices=max_u, extra_passes=extra_u
            )

        space_bev = cv2.bitwise_and(space_bev, union_bev)
        lot_bev = union_bev

        bev = np.zeros((h, w, 3), np.uint8)
        bev[lot_bev > 0] = (255, 255, 255)
        bev[space_bev > 0] = (255, 0, 0)
        bev[out_bev > 0] = (0, 0, 255)

        # === ADD: draw BEV cut boundary due to perspective warp ===
        self._draw_bev_cut_boundary(bev, M, w, h)

        if bool(self.get_parameter("draw_simplified_edges").value):
            self._draw_polys(bev, lot_polys, (0, 255, 255), thickness=2)
            self._draw_polys(bev, space_polys, (0, 255, 0), thickness=2)
            self._draw_polys(bev, union_polys, (255, 0, 255), thickness=2)

        out_msg = self.bridge.cv2_to_imgmsg(bev, encoding="bgr8")
        out_msg.header = img_msg.header
        self.viz_pub.publish(out_msg)


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
