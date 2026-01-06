#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion

import yaml


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


def as_points_xy(xy_list: List[List[float]], z: float = 0.0) -> List[Point]:
    pts: List[Point] = []
    for x, y in xy_list:
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        pts.append(p)
    return pts


def close_loop(points: List[Point]) -> List[Point]:
    if not points:
        return points
    out = list(points)
    out.append(out[0])
    return out


class ParkingMapVizNode(Node):
    def __init__(self) -> None:
        super().__init__("parking_map_viz")

        default_yaml = "/home/sg/gyeonggi_ws/src/decision_making_pkg/decision_making_pkg/config/parking_map.yaml"

        # Parameters (launch에서 모두 덮어쓸 수 있음)
        self.declare_parameter("yaml_path", default_yaml)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("topic", "/parking_map/markers")

        self.declare_parameter("z", 0.05)
        self.declare_parameter("line_width", 0.06)
        self.declare_parameter("slot_line_width", 0.05)
        self.declare_parameter("text_size", 0.25)
        self.declare_parameter("text_z_offset", 0.35)

        # On/Off & Filter
        self.declare_parameter("show_parking_lot", True)
        self.declare_parameter("selected_slot_id", "slot_02")
        self.declare_parameter("show_slot_labels", False)
        self.declare_parameter("show_target_labels", False)

        # Internal states (parameter callback로 갱신됨)
        self.yaml_path = ""
        self.frame_id = ""
        self.topic = ""

        self.z = 0.05
        self.line_width = 0.06
        self.slot_line_width = 0.05
        self.text_size = 0.25
        self.text_z_offset = 0.35

        self.show_parking_lot = True
        self.selected_slot_id = ""
        self.show_slot_labels = False
        self.show_target_labels = False

        # Apply initial parameter values
        self._apply_params_from_server()

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(MarkerArray, self.topic, qos)

        self.map_data: Dict[str, Any] = {}
        self._load_yaml()

        # parameter update callback (런타임 변경 반영)
        self.add_on_set_parameters_callback(self._on_params)

        self.timer = self.create_timer(0.5, self._on_timer)

        self.get_logger().info(f"YAML: {self.yaml_path}")
        self.get_logger().info(f"Publishing: {self.topic} (frame_id={self.frame_id})")
        self.get_logger().info(
            f"show_parking_lot={self.show_parking_lot}, selected_slot_id='{self.selected_slot_id}', "
            f"show_slot_labels={self.show_slot_labels}, show_target_labels={self.show_target_labels}"
        )

    def _apply_params_from_server(self) -> None:
        self.yaml_path = self.get_parameter("yaml_path").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.topic = self.get_parameter("topic").get_parameter_value().string_value

        self.z = float(self.get_parameter("z").value)
        self.line_width = float(self.get_parameter("line_width").value)
        self.slot_line_width = float(self.get_parameter("slot_line_width").value)
        self.text_size = float(self.get_parameter("text_size").value)
        self.text_z_offset = float(self.get_parameter("text_z_offset").value)

        self.show_parking_lot = bool(self.get_parameter("show_parking_lot").value)
        self.selected_slot_id = str(self.get_parameter("selected_slot_id").value).strip()
        self.show_slot_labels = bool(self.get_parameter("show_slot_labels").value)
        self.show_target_labels = bool(self.get_parameter("show_target_labels").value)

    def _on_params(self, params) -> SetParametersResult:
        # Validate
        for p in params:
            if p.name == "selected_slot_id" and p.type_ == p.Type.STRING:
                # allow "" or "slot_XX"
                pass
            if p.name in ("z", "line_width", "slot_line_width", "text_size", "text_z_offset"):
                if p.value is not None and float(p.value) < 0.0:
                    return SetParametersResult(successful=False, reason=f"{p.name} must be >= 0")

        # Apply changes
        reload_yaml = False
        for p in params:
            if p.name == "yaml_path":
                self.yaml_path = str(p.value)
                reload_yaml = True
            elif p.name == "frame_id":
                self.frame_id = str(p.value)
            elif p.name == "topic":
                # topic 변경은 publisher 재생성이 필요해서 여기서는 막는 게 안전함
                return SetParametersResult(successful=False, reason="topic cannot be changed at runtime")
            elif p.name == "z":
                self.z = float(p.value)
            elif p.name == "line_width":
                self.line_width = float(p.value)
            elif p.name == "slot_line_width":
                self.slot_line_width = float(p.value)
            elif p.name == "text_size":
                self.text_size = float(p.value)
            elif p.name == "text_z_offset":
                self.text_z_offset = float(p.value)
            elif p.name == "show_parking_lot":
                self.show_parking_lot = bool(p.value)
            elif p.name == "selected_slot_id":
                self.selected_slot_id = str(p.value).strip()
            elif p.name == "show_slot_labels":
                self.show_slot_labels = bool(p.value)
            elif p.name == "show_target_labels":
                self.show_target_labels = bool(p.value)

        if reload_yaml:
            self._load_yaml()

        return SetParametersResult(successful=True)

    def _load_yaml(self) -> None:
        p = Path(self.yaml_path).expanduser().resolve()
        self.yaml_path = str(p)

        if not p.exists():
            self.get_logger().error(f"YAML not found: {self.yaml_path}")
            self.map_data = {}
            return

        with p.open("r", encoding="utf-8") as f:
            self.map_data = yaml.safe_load(f) or {}

    def _mk_line_strip(
        self,
        mid: int,
        ns: str,
        points: List[Point],
        rgba: Tuple[float, float, float, float],
        width: float,
    ) -> Marker:
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = float(width)
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.points = points
        m.lifetime.sec = 0
        return m

    def _mk_text(
        self,
        mid: int,
        ns: str,
        x: float,
        y: float,
        text: str,
        rgba: Tuple[float, float, float, float],
        size: float,
    ) -> Marker:
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(self.z + self.text_z_offset)
        m.pose.orientation.w = 1.0
        m.scale.z = float(size)
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.text = text
        m.lifetime.sec = 0
        return m

    def _slot_selected(self, slot_id: str) -> bool:
        if self.selected_slot_id == "":
            return True
        return slot_id == self.selected_slot_id

    def _build_markers(self) -> MarkerArray:
        arr = MarkerArray()
        if not self.map_data:
            return arr

        mid = 0

        if self.show_parking_lot:
            parking_lot = self.map_data.get("parking_lot", {})
            corners_xy = parking_lot.get("corners_xy", [])
            if corners_xy:
                pts = close_loop(as_points_xy(corners_xy, z=self.z))
                arr.markers.append(self._mk_line_strip(mid, "parking_lot", pts, (1.0, 1.0, 1.0, 1.0), self.line_width))
                mid += 1

        in_line = self.map_data.get("in_line", {})
        p1 = in_line.get("p1_xy", None)
        p2 = in_line.get("p2_xy", None)
        if p1 and p2:
            pts = as_points_xy([p1, p2], z=self.z)
            arr.markers.append(self._mk_line_strip(mid, "in_line", pts, (0.2, 1.0, 0.2, 1.0), self.line_width))
            mid += 1

        out_line = self.map_data.get("out_line", {})
        p1 = out_line.get("p1_xy", None)
        p2 = out_line.get("p2_xy", None)
        if p1 and p2:
            pts = as_points_xy([p1, p2], z=self.z)
            arr.markers.append(self._mk_line_strip(mid, "out_line", pts, (1.0, 0.2, 0.2, 1.0), self.line_width))
            mid += 1

        slots = self.map_data.get("slots", [])
        for s in slots:
            sid = str(s.get("id", "slot"))
            if not self._slot_selected(sid):
                continue

            scorners = s.get("corners_xy", [])
            if scorners:
                pts = close_loop(as_points_xy(scorners, z=self.z))
                arr.markers.append(self._mk_line_strip(mid, "slots", pts, (0.2, 0.4, 1.0, 1.0), self.slot_line_width))
                mid += 1

                if self.show_slot_labels:
                    cx = sum([p[0] for p in scorners]) / len(scorners)
                    cy = sum([p[1] for p in scorners]) / len(scorners)
                    arr.markers.append(self._mk_text(mid, "slot_labels", cx, cy, sid, (0.9, 0.9, 0.9, 1.0), self.text_size))
                    mid += 1

            tpose = s.get("target_pose", None)
            if tpose and len(tpose) >= 3 and self.show_target_labels:
                tx, ty = float(tpose[0]), float(tpose[1])
                arr.markers.append(self._mk_text(mid, "target_pose_label", tx, ty, f"{sid}_target", (1.0, 0.9, 0.1, 1.0), self.text_size))
                mid += 1

        return arr

    def _on_timer(self) -> None:
        self.pub.publish(self._build_markers())


def main() -> None:
    rclpy.init()
    node = ParkingMapVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
