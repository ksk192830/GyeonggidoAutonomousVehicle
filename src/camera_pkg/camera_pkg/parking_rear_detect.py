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
from interfaces_pkg.msg import DetectionArray, EndLine, ParkingSpace
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

import cv2
import numpy as np
import math


class ParkingRearDetect(Node):
    def __init__(self):
        super().__init__("parking_rear_detect")

        self.declare_parameter("image_topic", "/cam1/image_raw")
        self.declare_parameter("detection_topic", "/cam1/detections")
        self.declare_parameter("viz_topic", "/rear_viz")

        self.declare_parameter(
            "persp_src",
            [0.0, 0.42, 
    		 1.0, 0.42, 
    		 0.86, 0.3, 
    		 0.14, 0.3 ]
        )
        self.declare_parameter(
            "persp_dst",
            [0.12, 0.56,
    		 0.88, 0.56, 
    		 0.86, 0.3, 
    		 0.14, 0.3 ]
        )

        self.declare_parameter("morph_kernel", 7)
        self.declare_parameter("slop", 1.0)
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

        self.declare_parameter("draw_bev_cut_boundary", True)
        self.declare_parameter("bev_cut_boundary_thickness", 2)
        self.declare_parameter("bev_cut_boundary_color_bgr", [0, 255, 255])
        self.declare_parameter("bev_cut_boundary_min_y_ratio", 0.5)

        self.declare_parameter("bg_enable", True)
        self.declare_parameter("bg_opacity", 0.55)

        self.declare_parameter("enable_parking_lot", False)
        self.declare_parameter("enable_parking_space", True)
        self.declare_parameter("enable_end_line", True)
        
        # Fixed parking space width in pixels for BEV
        self.declare_parameter("parking_width_px", 528.0)

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

        self.ts = ApproximateTimeSynchronizer([img_sub, det_sub], queue_size=10, slop=slop)
        self.ts.registerCallback(self.sync_callback)

        self.viz_pub = self.create_publisher(Image, viz_topic, qos)
        self.end_line_pub = self.create_publisher(EndLine, "/rear_end_line", 10)
        self.parking_space_pub = self.create_publisher(ParkingSpace, "/rear_parking_space", 10)
        
        self.get_logger().info(
            f"parking_rear_detect initialized. Sub: {image_topic}, {detection_topic} -> Pub: {viz_topic}, /rear_end_line, /rear_parking_space"
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

        cv2.rectangle(bev_bgr, (0, 0), (w - 1, h - 1), color, thickness)

    def _make_bev_background(self, frame_bgr: np.ndarray, M: np.ndarray, w: int, h: int) -> np.ndarray:
        bg = cv2.warpPerspective(
            frame_bgr,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return bg

    @staticmethod
    def _alpha_blend(base_bgr: np.ndarray, overlay_bgr: np.ndarray, opacity: float) -> np.ndarray:
        a = float(np.clip(opacity, 0.0, 1.0))
        if a <= 0.0:
            return overlay_bgr
        if a >= 1.0:
            return base_bgr
        return cv2.addWeighted(base_bgr, a, overlay_bgr, 1.0 - a, 0.0)

    @staticmethod
    def _scale_contour(cnt, scale):
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return cnt
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        return cnt_scaled.astype(np.int32)

    def _detect_parking_center_line(self, frame_bev, roi_mask, overlay):
        """
        Detect parking lines and calculate center line based on scenarios:
        1. Right line only -> Shift Left by half width
        2. Both lines -> Average
        3. Left line only -> Shift Right by half width (fallback)
        """
        msg = ParkingSpace()
        msg.found = False
        
        h, w = frame_bev.shape[:2]
        target_width = float(self.get_parameter("parking_width_px").value)
        half_width = target_width / 2.0
        
        # 1. Pre-processing
        gray = cv2.cvtColor(frame_bev, cv2.COLOR_BGR2GRAY)
        
        # Create a mask for valid image area (non-black) to ignore BEV borders
        _, valid_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # Erode the valid mask slightly to remove the boundary edge itself
        valid_mask = cv2.erode(valid_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        
        # Mask edges: Keep edges only inside the valid image area AND inside the ROI
        final_mask = cv2.bitwise_and(valid_mask, roi_mask)
        masked_edges = cv2.bitwise_and(edges, edges, mask=final_mask)
        
        # 2. Hough Transform to find lines
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=20
        )
        
        left_lines = []
        right_lines = []
        
        cx = w // 2
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Filter out horizontal-ish lines
                if abs(x2 - x1) > abs(y2 - y1) * 2: # Very horizontal
                    continue
                    
                # Calculate slope and midpoint
                slope = (x2 - x1) / (y2 - y1 + 1e-6)
                mid_x = (x1 + x2) / 2
                
                # Classify based on position
                if mid_x < cx:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))

        # Visualize detected segments for debugging
        for l in left_lines:
            cv2.line(overlay, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2) # Green
        for r in right_lines:
            cv2.line(overlay, (r[0], r[1]), (r[2], r[3]), (0, 255, 255), 2) # Yellow
                    
        # Helper to fit a single line from segments
        def fit_single_line(line_segments):
            if not line_segments:
                return None
            pts = []
            for l in line_segments:
                pts.append([l[0], l[1]])
                pts.append([l[2], l[3]])
            pts = np.array(pts, dtype=np.int32)
            
            # fitLine returns normalized vector (vx, vy) and point (x0, y0)
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            return (float(vx), float(vy), float(x0), float(y0))

        l_param = fit_single_line(left_lines)
        r_param = fit_single_line(right_lines)
        
        center_line = None # (vx, vy, x0, y0)
        
        # Logic Selection
        if l_param is not None and r_param is not None:
            # Case 3: Both lines visible -> Average
            lvx, lvy, lx0, ly0 = l_param
            rvx, rvy, rx0, ry0 = r_param
            
            avg_vx = (lvx + rvx) / 2
            avg_vy = (lvy + rvy) / 2
            avg_x0 = (lx0 + rx0) / 2
            avg_y0 = (ly0 + ry0) / 2
            center_line = (avg_vx, avg_vy, avg_x0, avg_y0)
            
        elif r_param is not None:
            # Case 1 & 2: Right line only -> Shift Left
            vx, vy, x0, y0 = r_param
            
            # Assuming lines are roughly vertical.
            # If vy > 0 (down), "Left" is -x direction relative to line.
            # Ideally: perpendicular shift.
            # Normal vector to (vx, vy) is (-vy, vx).
            # If we assume (vx, vy) points DOWN, left is positive x in normal frame?
            # Let's stick to simple horizontal shift for robustness as parking lines are vertical.
            center_line = (vx, vy, x0 - half_width, y0)
            
        elif l_param is not None:
            # Case 4: Left line only -> Shift Right
            vx, vy, x0, y0 = l_param
            center_line = (vx, vy, x0 + half_width, y0)
            
        else:
            return msg # Nothing found

        # Calculate Output & Visualize
        vx, vy, x0, y0 = center_line
        
        # Avoid division by zero
        if abs(vy) < 1e-3: 
             return msg
             
        # Find x intersection with bottom (y=h)
        # (x - x0)/vx = (y - y0)/vy  => x = x0 + (vx/vy)*(y-y0)
        x_bottom = x0 + (vx/vy) * (h - y0)
        
        # Find x intersection with top (y=0) for visualization
        x_top = x0 + (vx/vy) * (0 - y0)
        
        # Yaw calculation (angle with vertical)
        # atan2(vy, vx) is angle of vector.
        # We usually want 0 for vertical up, or vertical down.
        # Let's use standard yaw convention: atan2(dy, dx)
        # But car control usually expects error angle.
        final_yaw = math.atan2(vy, vx)
        
        # Normalize yaw to be consistent?
        # If line points down (vy>0), yaw is ~90 deg (pi/2).
        # If vertical up (vy<0), yaw is ~-90 deg (-pi/2).
        
        # Visualize Center Line (Thick White)
        pt1 = (int(x_top), 0)
        pt2 = (int(x_bottom), h)
        cv2.line(overlay, pt1, pt2, (255, 255, 255), 5)
        
        msg.found = True
        msg.x = float(x_bottom / w) # Normalize 0~1
        msg.yaw = float(final_yaw)
        
        return msg

    def sync_callback(self, img_msg: Image, det_msg: DetectionArray):
        self.get_logger().info("Sync callback entered")
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

        en_lot = bool(self.get_parameter("enable_parking_lot").value)
        en_space = bool(self.get_parameter("enable_parking_space").value)
        en_end = bool(self.get_parameter("enable_end_line").value)

        lot_mask = np.zeros((h, w), np.uint8)
        space_mask = np.zeros((h, w), np.uint8)
        end_mask = np.zeros((h, w), np.uint8)

        for det in det_msg.detections:
            cls = getattr(det, "class_name", "")
            if not hasattr(det, "mask") or det.mask is None:
                continue
            if not hasattr(det.mask, "data") or det.mask.data is None:
                continue

            if cls == "parking_lot" and not en_lot:
                continue
            if cls == "parking_space" and not en_space:
                continue
            if cls == "end_line" and not en_end:
                continue

            m = self._poly_to_mask(det.mask.data, h, w)
            if m is None or m.size == 0:
                continue

            if cls == "parking_lot":
                lot_mask = cv2.bitwise_or(lot_mask, m)
            elif cls == "parking_space":
                space_mask = cv2.bitwise_or(space_mask, m)
            elif cls == "end_line":
                end_mask = cv2.bitwise_or(end_mask, m)

        lot_bev = cv2.warpPerspective(lot_mask, M, (w, h), flags=cv2.INTER_NEAREST)
        space_bev = cv2.warpPerspective(space_mask, M, (w, h), flags=cv2.INTER_NEAREST)
        end_bev = cv2.warpPerspective(end_mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Warp original frame for edge detection
        frame_bev = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)

        if en_lot:
            lot_bev = self._postprocess_mask(lot_bev)
        else:
            lot_bev[:] = 0

        space_roi = np.zeros((h, w), np.uint8)
        if en_space:
            space_bev = self._postprocess_mask(space_bev)
            
            # Fixed 15-pixel dilation for ROI
            kernel = np.ones((15, 15), np.uint8)
            space_roi = cv2.dilate(space_bev, kernel, iterations=1)
        else:
            space_bev[:] = 0
            
        if en_end:
            end_bev = self._postprocess_mask(end_bev)
        else:
            end_bev[:] = 0

        lot_polys = []
        space_polys = []
        union_polys = []

        if bool(self.get_parameter("simplify_enable").value):
            eps_space = float(self.get_parameter("simplify_eps_ratio_space").value)
            eps_lot = float(self.get_parameter("simplify_eps_ratio_lot").value)
            max_v = int(self.get_parameter("simplify_max_vertices").value)
            extra = int(self.get_parameter("simplify_extra_passes").value)

            if en_lot:
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

            if en_space:
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
            
            # We use space_roi for detection, but space_bev for visualization overlay logic?
            # Actually, let's keep space_bev as the original mask for visualization logic
            # But update the detection call below.
            pass 

        union_bev = self._join_masks(lot_bev, space_bev)

        if bool(self.get_parameter("union_simplify_enable").value):
            eps_u = float(self.get_parameter("union_eps_ratio").value)
            max_u = int(self.get_parameter("union_max_vertices").value)
            extra_u = int(self.get_parameter("union_extra_passes").value)
            union_bev, union_polys = self._simplify_to_polygon_mask(
                union_bev, img_area=img_area, eps_ratio=eps_u, max_vertices=max_u, extra_passes=extra_u
            )

        if en_space:
            space_bev = cv2.bitwise_and(space_bev, union_bev)
        else:
            space_bev[:] = 0

        if en_lot:
            lot_bev = union_bev
        else:
            lot_bev[:] = 0

        overlay = np.zeros((h, w, 3), np.uint8)
        
        if en_lot:
            overlay[lot_bev > 0] = (255, 255, 255)
        if en_space:
            # Visualize the 1.2x dilated region (space_roi) instead of the original mask
            overlay[space_roi > 0] = (255, 0, 0)
        if en_end:
            overlay[end_bev > 0] = (0, 0, 255)

        self._draw_bev_cut_boundary(overlay, M, w, h)

        if bool(self.get_parameter("draw_simplified_edges").value):
            if en_lot:
                self._draw_polys(overlay, lot_polys, (0, 255, 255), thickness=2)
            if en_space:
                self._draw_polys(overlay, space_polys, (0, 255, 0), thickness=2)
            if en_lot or en_space:
                self._draw_polys(overlay, union_polys, (255, 0, 255), thickness=2)

        # --- Parking Space Center Line ---
        space_msg = ParkingSpace()
        space_msg.found = False
        if en_space:
            try:
                 # Pass 'frame_bev' (image), 'roi_mask' (dilated mask), and 'overlay'
                 # Note: 'space_roi' was defined earlier as dilated mask
                 space_msg = self._detect_parking_center_line(frame_bev, space_roi, overlay)
            except Exception as e:
                self.get_logger().warn(f"Space center calculation failed: {e}")
        
        self.parking_space_pub.publish(space_msg)
        # ---------------------------------

        # --- EndLine Calculation & Visualization (Simple & Robust) ---
        end_msg = EndLine()
        end_msg.found = False
        
        # end_bev is valid if en_end is True.
        if en_end:
            try:
                cnts, _ = cv2.findContours(end_bev, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(c) > 50:
                        # 1. Get all points
                        points = c[:, 0, :]
                        
                        # 2. Robust Direction: Fit line to ALL points
                        # This gives the average slope of the entire blob
                        line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                        vx, vy, _, _ = line_params.flatten()
                        
                        # Normalize direction (pointing right)
                        if vx < 0:
                            vx, vy = -vx, -vy
                        
                        # 3. Anchor Point: The bottom-most point (Max Y)
                        # Sort by Y descending, take the first one
                        sorted_indices = np.argsort(points[:, 1])[::-1]
                        bottom_point = points[sorted_indices[0]]
                        bx, by = bottom_point
                        
                        # 4. Calculate Yaw
                        yaw = math.atan2(vy, vx)
                        
                        # 5. Publish
                        end_msg.found = True
                        end_msg.x = float(bx)
                        end_msg.y = float(by)
                        end_msg.yaw = float(yaw)
                        
                        # 6. Visualization
                        # Draw line passing through (bx, by) with slope (vy/vx)
                        # y = tan(yaw) * (x - bx) + by
                        tan_yaw = vy / (vx + 1e-6) # Avoid division by zero
                        
                        y_at_0 = tan_yaw * (0 - bx) + by
                        y_at_w = tan_yaw * (w - bx) + by
                        
                        cv2.line(overlay, (0, int(y_at_0)), (w, int(y_at_w)), (0, 255, 0), 10)

            except Exception as e:
                self.get_logger().warn(f"EndLine calculation failed: {e}")
        else:
            pass

        self.end_line_pub.publish(end_msg)

        bg_enable = bool(self.get_parameter("bg_enable").value)
        bg_opacity = float(self.get_parameter("bg_opacity").value)

        if bg_enable and bg_opacity > 0.001:
            bev_bg = self._make_bev_background(frame, M, w, h)
            bev = self._alpha_blend(bev_bg, overlay, bg_opacity)
        else:
            bev = overlay

        out_msg = self.bridge.cv2_to_imgmsg(bev, encoding="bgr8")
        out_msg.header = img_msg.header
        self.viz_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ParkingRearDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()