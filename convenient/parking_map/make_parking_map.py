#!/usr/bin/env python3
import os
import sys
import yaml
import math
import cv2
import numpy as np

WINDOW = "parking_map_builder"

STATE_ORIGIN = "origin"
STATE_SCALE_P1 = "scale_p1"
STATE_SCALE_P2 = "scale_p2"
STATE_LOT = "parking_lot"
STATE_IN_LINE_P1 = "in_line_p1"
STATE_IN_LINE_P2 = "in_line_p2"
STATE_SLOTS = "slots"
STATE_OUT_LINE_P1 = "out_line_p1"
STATE_OUT_LINE_P2 = "out_line_p2"

REQUIRED_SLOTS = 4


def dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def px_to_map(u, v, origin_uv, m_per_px):
    u0, v0 = origin_uv
    x = (u - u0) * m_per_px
    y = (v0 - v) * m_per_px
    return [float(x), float(y)]


def order_quad(points):
    pts = np.array(points, dtype=np.float32)
    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(angles)
    pts = pts[idx]
    return pts.tolist()


def quad_center(quad_xy):
    q = np.array(quad_xy, dtype=np.float32)
    c = np.mean(q, axis=0)
    return [float(c[0]), float(c[1])]


def yaw_from_quad(quad_xy):
    q = np.array(quad_xy, dtype=np.float32)
    edges = [
        (q[1] - q[0], np.linalg.norm(q[1] - q[0])),
        (q[2] - q[1], np.linalg.norm(q[2] - q[1])),
        (q[3] - q[2], np.linalg.norm(q[3] - q[2])),
        (q[0] - q[3], np.linalg.norm(q[0] - q[3])),
    ]
    vec, _ = max(edges, key=lambda e: e[1])
    yaw = math.atan2(float(vec[1]), float(vec[0]))
    return float(yaw)


class Builder:
    def __init__(
        self,
        image_path: str,
        out_yaml: str = "parking_map.yaml",
        scale_mm: float = 1500.0,
        lot_w_mm: float = 4000.0,
        lot_h_mm: float = 1550.0,
        slot_w_mm: float = 1500.0,
        slot_h_mm: float = 950.0,
        top_margin_px: int = 180,
    ):
        self.image_path = image_path
        self.out_yaml = out_yaml

        self.scale_mm = float(scale_mm)
        self.lot_w_mm = float(lot_w_mm)
        self.lot_h_mm = float(lot_h_mm)
        self.slot_w_mm = float(slot_w_mm)
        self.slot_h_mm = float(slot_h_mm)

        self.top_margin_px = int(top_margin_px)

        self.map_img0 = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.map_img0 is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        self.reset_all()

    def reset_all(self):
        self.state = STATE_ORIGIN

        self.origin_uv = None

        self.scale_p1 = None
        self.scale_p2 = None
        self.m_per_px = None

        self.lot_pts_uv = []

        self.in_line_p1 = None
        self.in_line_p2 = None

        self.current_slot_pts = []
        self.slots = []
        self.slot_idx = 1

        self.out_line_p1 = None
        self.out_line_p2 = None

    def _make_canvas(self):
        h, w = self.map_img0.shape[:2]
        canvas = np.zeros((h + self.top_margin_px, w, 3), dtype=np.uint8)
        canvas[: self.top_margin_px, :] = (20, 20, 20)
        canvas[self.top_margin_px :, :] = self.map_img0
        return canvas

    def _draw_text_in_margin(self, canvas, lines):
        x0 = 20
        y0 = 35
        dy = 28
        for i, line in enumerate(lines[:7]):
            y = y0 + i * dy
            cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 1, cv2.LINE_AA)

    def _draw_point_uv(self, canvas, p_uv, color=(0, 255, 255), r=6):
        u, v = int(p_uv[0]), int(p_uv[1])
        cv2.circle(canvas, (u, v + self.top_margin_px), r, color, -1, cv2.LINE_AA)

    def _draw_line_uv(self, canvas, p1_uv, p2_uv, color=(0, 255, 0), t=2):
        u1, v1 = int(p1_uv[0]), int(p1_uv[1])
        u2, v2 = int(p2_uv[0]), int(p2_uv[1])
        cv2.line(canvas, (u1, v1 + self.top_margin_px), (u2, v2 + self.top_margin_px), color, t, cv2.LINE_AA)

    def _draw_poly_uv(self, canvas, pts_uv, color=(255, 0, 255), t=2):
        if len(pts_uv) < 2:
            return
        pts = np.array([[int(u), int(v + self.top_margin_px)] for u, v in pts_uv], dtype=np.int32).reshape(-1, 1, 2)
        closed = len(pts_uv) >= 3
        cv2.polylines(canvas, [pts], closed, color, t, cv2.LINE_AA)

    def _status_lines(self):
        lines = []
        if self.state == STATE_ORIGIN:
            lines.append("Step 1/7: Click origin (map 0,0).")
            lines.append("Keys: s save, u undo, r reset, q quit")
        elif self.state == STATE_SCALE_P1:
            lines.append(f"Step 2/7: Click scale point 1 for {int(self.scale_mm)}mm segment.")
            lines.append("Pick a segment labeled 1500mm.")
        elif self.state == STATE_SCALE_P2:
            lines.append("Step 2/7: Click scale point 2 (end of the same segment).")
        elif self.state == STATE_LOT:
            lines.append("Step 3/7: Click 4 corners of parking_lot.")
            lines.append(f"lot_size_mm: {int(self.lot_w_mm)} x {int(self.lot_h_mm)}")
        elif self.state == STATE_IN_LINE_P1:
            lines.append("Step 4/7: Click in_line point 1 (1500mm).")
        elif self.state == STATE_IN_LINE_P2:
            lines.append("Step 4/7: Click in_line point 2 (end).")
        elif self.state == STATE_SLOTS:
            remaining = REQUIRED_SLOTS - len(self.slots)
            lines.append(f"Step 5/7: Click 4 corners per slot. Need {remaining} more slot(s).")
            lines.append(f"slot_size_mm: {int(self.slot_w_mm)} x {int(self.slot_h_mm)}")
            lines.append("Keys: n cancel slot, u undo, r reset, s save")
        elif self.state == STATE_OUT_LINE_P1:
            lines.append("Step 6/7: Click out_line point 1 (1500mm).")
        elif self.state == STATE_OUT_LINE_P2:
            lines.append("Step 6/7: Click out_line point 2 (end).")
            lines.append("Step 7/7: Press 's' to save YAML.")
        else:
            lines.append("Unknown state.")

        if self.m_per_px is not None:
            lines.append(f"meters_per_pixel: {self.m_per_px:.6f}")

        return lines

    def on_click(self, x, y_canvas):
        y_map = y_canvas - self.top_margin_px
        if y_map < 0:
            return
        p = [float(x), float(y_map)]

        if self.state == STATE_ORIGIN:
            self.origin_uv = p
            self.state = STATE_SCALE_P1
            return

        if self.state == STATE_SCALE_P1:
            self.scale_p1 = p
            self.state = STATE_SCALE_P2
            return

        if self.state == STATE_SCALE_P2:
            self.scale_p2 = p
            px_len = dist(self.scale_p1, self.scale_p2)
            if px_len < 1e-6:
                self.scale_p2 = None
                return
            self.m_per_px = (self.scale_mm / 1000.0) / px_len
            self.state = STATE_LOT
            return

        if self.state == STATE_LOT:
            self.lot_pts_uv.append(p)
            if len(self.lot_pts_uv) == 4:
                self.lot_pts_uv = order_quad(self.lot_pts_uv)
                self.state = STATE_IN_LINE_P1
            return

        if self.state == STATE_IN_LINE_P1:
            self.in_line_p1 = p
            self.state = STATE_IN_LINE_P2
            return

        if self.state == STATE_IN_LINE_P2:
            self.in_line_p2 = p
            self.state = STATE_SLOTS
            return

        if self.state == STATE_SLOTS:
            self.current_slot_pts.append(p)
            if len(self.current_slot_pts) == 4:
                quad_uv = order_quad(self.current_slot_pts)
                quad_xy = [px_to_map(u, v, self.origin_uv, self.m_per_px) for u, v in quad_uv]
                center = quad_center(quad_xy)
                yaw = yaw_from_quad(quad_xy)

                slot = {
                    "id": f"slot_{self.slot_idx:02d}",
                    "size_mm": [self.slot_w_mm, self.slot_h_mm],
                    "corners_uv": [[float(u), float(v)] for u, v in quad_uv],
                    "corners_xy": quad_xy,
                    "target_pose": [center[0], center[1], yaw],
                }
                self.slots.append(slot)
                self.slot_idx += 1
                self.current_slot_pts = []

                if len(self.slots) >= REQUIRED_SLOTS:
                    self.state = STATE_OUT_LINE_P1
            return

        if self.state == STATE_OUT_LINE_P1:
            self.out_line_p1 = p
            self.state = STATE_OUT_LINE_P2
            return

        if self.state == STATE_OUT_LINE_P2:
            self.out_line_p2 = p
            return

    def cancel_current_slot(self):
        self.current_slot_pts = []

    def undo(self):
        if self.state == STATE_OUT_LINE_P2:
            if self.out_line_p2 is not None:
                self.out_line_p2 = None
            else:
                self.state = STATE_OUT_LINE_P1
            return

        if self.state == STATE_OUT_LINE_P1:
            if self.out_line_p1 is not None:
                self.out_line_p1 = None
                self.state = STATE_SLOTS
            return

        if self.state == STATE_SLOTS:
            if self.current_slot_pts:
                self.current_slot_pts.pop()
                return
            if self.slots:
                self.slots.pop()
                self.slot_idx = max(1, self.slot_idx - 1)
                self.state = STATE_SLOTS
                return
            self.state = STATE_IN_LINE_P2
            return

        if self.state == STATE_IN_LINE_P2:
            if self.in_line_p2 is not None:
                self.in_line_p2 = None
                return
            self.state = STATE_IN_LINE_P1
            return

        if self.state == STATE_IN_LINE_P1:
            if self.in_line_p1 is not None:
                self.in_line_p1 = None
                return
            self.state = STATE_LOT
            return

        if self.state == STATE_LOT:
            if self.lot_pts_uv:
                self.lot_pts_uv.pop()
                return
            self.state = STATE_SCALE_P2
            return

        if self.state == STATE_SCALE_P2:
            if self.scale_p2 is not None:
                self.scale_p2 = None
                self.m_per_px = None
                self.state = STATE_SCALE_P1
                return
            self.state = STATE_SCALE_P1
            return

        if self.state == STATE_SCALE_P1:
            if self.scale_p1 is not None:
                self.scale_p1 = None
                self.state = STATE_ORIGIN
                return
            self.state = STATE_ORIGIN
            return

        if self.state == STATE_ORIGIN:
            self.origin_uv = None

    def save_yaml(self):
        if self.origin_uv is None:
            print("origin not set")
            return
        if self.m_per_px is None:
            print("scale not set")
            return
        if len(self.lot_pts_uv) != 4:
            print("parking_lot not set (need 4 corners)")
            return
        if self.in_line_p1 is None or self.in_line_p2 is None:
            print("in_line not set")
            return
        if len(self.slots) != REQUIRED_SLOTS:
            print(f"need exactly {REQUIRED_SLOTS} slots, current={len(self.slots)}")
            return
        if self.out_line_p1 is None or self.out_line_p2 is None:
            print("out_line not set")
            return

        lot_xy = [px_to_map(u, v, self.origin_uv, self.m_per_px) for u, v in self.lot_pts_uv]

        in_line_uv = [self.in_line_p1, self.in_line_p2]
        out_line_uv = [self.out_line_p1, self.out_line_p2]
        in_line_xy = [px_to_map(u, v, self.origin_uv, self.m_per_px) for u, v in in_line_uv]
        out_line_xy = [px_to_map(u, v, self.origin_uv, self.m_per_px) for u, v in out_line_uv]

        data = {
            "image": {
                "path": os.path.basename(self.image_path),
                "width": int(self.map_img0.shape[1]),
                "height": int(self.map_img0.shape[0]),
            },
            "map_frame": {
                "origin_uv": [float(self.origin_uv[0]), float(self.origin_uv[1])],
                "meters_per_pixel": float(self.m_per_px),
                "axes": {"x": "right", "y": "up", "yaw0": "facing +x"},
            },
            "geometry_mm": {
                "parking_lot_size": [self.lot_w_mm, self.lot_h_mm],
                "slot_size": [self.slot_w_mm, self.slot_h_mm],
                "line_length": float(self.scale_mm),
            },
            "parking_lot": {
                "size_mm": [self.lot_w_mm, self.lot_h_mm],
                "corners_uv": [[float(u), float(v)] for u, v in self.lot_pts_uv],
                "corners_xy": lot_xy,
            },
            "in_line": {
                "size_mm": float(self.scale_mm),
                "p1_uv": [float(self.in_line_p1[0]), float(self.in_line_p1[1])],
                "p2_uv": [float(self.in_line_p2[0]), float(self.in_line_p2[1])],
                "p1_xy": in_line_xy[0],
                "p2_xy": in_line_xy[1],
            },
            "slots": self.slots,
            "out_line": {
                "size_mm": float(self.scale_mm),
                "p1_uv": [float(self.out_line_p1[0]), float(self.out_line_p1[1])],
                "p2_uv": [float(self.out_line_p2[0]), float(self.out_line_p2[1])],
                "p1_xy": out_line_xy[0],
                "p2_xy": out_line_xy[1],
            },
        }

        with open(self.out_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        print(f"Saved: {self.out_yaml}")

    def render(self):
        canvas = self._make_canvas()
        self._draw_text_in_margin(canvas, self._status_lines())

        if self.origin_uv is not None:
            self._draw_point_uv(canvas, self.origin_uv, (0, 165, 255), 8)

        if self.scale_p1 is not None:
            self._draw_point_uv(canvas, self.scale_p1, (0, 255, 0), 6)
        if self.scale_p2 is not None:
            self._draw_point_uv(canvas, self.scale_p2, (0, 255, 0), 6)
            self._draw_line_uv(canvas, self.scale_p1, self.scale_p2, (0, 255, 0), 2)

        if self.lot_pts_uv:
            for p in self.lot_pts_uv:
                self._draw_point_uv(canvas, p, (255, 0, 255), 5)
            self._draw_poly_uv(canvas, self.lot_pts_uv, (255, 0, 255), 2)

        if self.in_line_p1 is not None:
            self._draw_point_uv(canvas, self.in_line_p1, (255, 255, 0), 6)
        if self.in_line_p2 is not None:
            self._draw_point_uv(canvas, self.in_line_p2, (255, 255, 0), 6)
            self._draw_line_uv(canvas, self.in_line_p1, self.in_line_p2, (255, 255, 0), 2)

        for s in self.slots:
            self._draw_poly_uv(canvas, s["corners_uv"], (255, 0, 255), 2)

        if self.current_slot_pts:
            for p in self.current_slot_pts:
                self._draw_point_uv(canvas, p, (255, 200, 0), 5)

        if self.out_line_p1 is not None:
            self._draw_point_uv(canvas, self.out_line_p1, (0, 0, 255), 6)
        if self.out_line_p2 is not None:
            self._draw_point_uv(canvas, self.out_line_p2, (0, 0, 255), 6)
            self._draw_line_uv(canvas, self.out_line_p1, self.out_line_p2, (0, 0, 255), 2)

        cv2.imshow(WINDOW, canvas)


def mouse_cb(event, x, y, flags, userdata):
    b = userdata
    if event == cv2.EVENT_LBUTTONDOWN:
        b.on_click(x, y)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 make_parking_map.py <map_image.png> [out_yaml]")
        sys.exit(1)

    image_path = sys.argv[1]
    out_yaml = sys.argv[2] if len(sys.argv) >= 3 else "parking_map.yaml"

    b = Builder(
        image_path=image_path,
        out_yaml=out_yaml,
        scale_mm=1500.0,
        lot_w_mm=4000.0,
        lot_h_mm=1550.0,
        slot_w_mm=1500.0,
        slot_h_mm=950.0,
        top_margin_px=180,
    )

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_cb, b)

    while True:
        b.render()
        k = cv2.waitKey(20) & 0xFF

        if k == ord("q") or k == 27:
            break
        if k == ord("u"):
            b.undo()
        if k == ord("r"):
            b.reset_all()
        if k == ord("n"):
            b.cancel_current_slot()
        if k == ord("s"):
            b.save_yaml()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
