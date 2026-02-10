#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy
)

from interfaces_pkg.msg import State

import math
import tf2_ros
import geometry_msgs.msg
import numpy as np

# A1 lidar control library (user-defined)
from .lib import lidar_perception_func_lib as LPFL

# ================= USER CONFIG =================
PUB_TOPIC_NAME = 'scan_raw'      # default publish topic name
LIDAR_PORT = '/dev/ttyUSB0'      # default lidar port

TIMER = 0.1                      # timer period [s]

FRAME_ID = 'laser_frame'
BASE_FRAME_ID = 'base_link'

# 0 ~ 360 deg in rad
ANGLE_MIN = 0.0
ANGLE_MAX = 2 * math.pi

# distance range [m]
RANGE_MIN = 0.0
RANGE_MAX = 4.0

# angle offset in degree
ANGLE_OFFSET_DEG = 180.0
# =================================================


class LidarScanNode(Node):
    def __init__(self):
        super().__init__('lidar_scan')

        # QoS configuration
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # Only parameters that are actually set from launch
        self.declare_parameter('pub_topic', PUB_TOPIC_NAME)
        self.declare_parameter('lidar_port', LIDAR_PORT)

        # scan publish 허용 state 목록
        self.declare_parameter("allowed_states", ["__ALL__"])
        self.allowed_states = list(self.get_parameter("allowed_states").get_parameter_value().string_array_value)
        self.current_state = None

        self.pub_topic = self.get_parameter('pub_topic').get_parameter_value().string_value
        self.lidar_port = self.get_parameter('lidar_port').get_parameter_value().string_value

        #Subscriber
        self.state_sub = self.create_subscription(State, "/motion_state", self.state_cb, 10)

        # Publisher
        self.publisher_ = self.create_publisher(LaserScan, self.pub_topic, qos_profile)


        # TF broadcaster (base_link -> laser_frame)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Internal lidar objects
        self.lidar = None
        self.lidar_sensor_data_generator = None

        # Initialize real lidar only
        self.get_logger().info("[REAL_LIDAR]")
        self.initialize_lidar()
        self.timer = self.create_timer(TIMER, self.publish_from_lidar)

    # ================= REAL MODE ONLY =================
    def initialize_lidar(self):
        """Initialize A1 lidar."""
        try:
            self.lidar = LPFL.RPLidar(self.lidar_port)
            self.lidar_sensor_data_generator = self.lidar.iter_scans()
            self.get_logger().info(f"LIDAR initialized on: {self.lidar_port}")
        except LPFL.RPLidarException as e:
            self.get_logger().error(f"LIDAR init failed: {e}")
            self.destroy_node()
            rclpy.shutdown()

    def reset_lidar(self):
        """Reset lidar connection."""
        try:
            if self.lidar is not None:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
        except LPFL.RPLidarException as e:
            self.get_logger().error(f"LIDAR reset failed: {e}")
        self.initialize_lidar()

    def publish_from_lidar(self):
        """Read from real lidar and publish LaserScan."""

        # broadcast TF
        self.broadcast_tf()

        # state가 있고, allowed_states에도 없으면 skip
        if self.current_state is not None:
            if "__ALL__" not in self.allowed_states and self.current_state not in self.allowed_states:
                return
            

        if self.lidar_sensor_data_generator is None:
            self.get_logger().error("LIDAR data generator is not initialized")
            return

        try:
            # scan: [(quality, angle_deg, distance_mm), ...]
            scan = next(self.lidar_sensor_data_generator)
            scan = np.array(scan)

            msg = self.build_laserscan_msg_from_measurements(scan)
            self.publisher_.publish(msg)
            # self.get_logger().info(f'Publishing: "{self.pub_topic}"')

        except StopIteration:
            self.get_logger().error("Failed to get LIDAR scan (StopIteration)")
            self.get_logger().error("❌" * 60) 
        except LPFL.RPLidarException as e:
            self.get_logger().error(f"RPLidar exception: {e}")
            self.get_logger().error("❌" * 60) 
            self.reset_lidar()
        except ValueError as e:
            self.get_logger().error(f"ValueError: {e}")
            self.get_logger().error("❌" * 60) 
            self.reset_lidar()

    def build_laserscan_msg_from_measurements(
        self,
        scan_np: np.ndarray
    ) -> LaserScan:
        """Convert RPLidar measurements to LaserScan message."""
        msg = LaserScan()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID

        msg.angle_min = float(ANGLE_MIN)
        msg.angle_max = float(ANGLE_MAX)
        msg.angle_increment = 2.0 * math.pi / 360.0  # 1 degree

        msg.time_increment = 0.0
        msg.scan_time = float(TIMER)
        msg.range_min = float(RANGE_MIN)
        msg.range_max = float(RANGE_MAX)

        num_points = int((msg.angle_max - msg.angle_min) / msg.angle_increment)
        ranges = [float('inf')] * num_points
        intensities = [0.0] * num_points

        for measurement in scan_np:
            # measurement: [quality, angle_deg, distance_mm]
            angle_deg_raw = measurement[1]
            dist_mm = measurement[2]
            quality = measurement[0]

            angle_deg = (angle_deg_raw + ANGLE_OFFSET_DEG) % 360.0

            # Drop measurements outside [60°, 300°]
            if angle_deg < 60.0 or angle_deg > 300.0:
                continue

            angle_rad = math.radians(angle_deg)

            if msg.angle_min <= angle_rad <= msg.angle_max:
                index = int((angle_rad - msg.angle_min) / msg.angle_increment)
                if 0 <= index < num_points:
                    distance_m = dist_mm / 1000.0
                    if RANGE_MIN <= distance_m <= RANGE_MAX:
                        ranges[index] = distance_m
                        intensities[index] = float(quality)
                    else:
                        ranges[index] = float('inf')
                        intensities[index] = 0.0


        msg.ranges = ranges
        msg.intensities = intensities

        return msg

    # ================= TF BROADCAST =================
    def broadcast_tf(self):
        """Publish base_link -> laser_frame transform."""
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = BASE_FRAME_ID
        t.child_frame_id = FRAME_ID

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    # ================= DESTRUCTOR =================
    def __del__(self):
        """Safe shutdown for lidar."""
        try:
            if self.lidar is not None:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
        except LPFL.RPLidarException as e:
            # Cannot use logger reliably in destructor, so just ignore
            pass
    
    def state_cb(self, msg: State):
        self.current_state = msg.state


def main(args=None):
    rclpy.init(args=args)
    node = LidarScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
