#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from interfaces_pkg.msg import ParkingSpace, MotionCommand, ParkingLot, EndLine, OutLine, State

import math

class ParkingNode(Node):

    def __init__(self):
        super().__init__("motion_parking")

        # Parameters
        self.margin = 7.0

        self.parking_line_position_weight = 135
        self.parking_line_angle_weight = 0.5

        self.parking_space_position_weight = 10
        self.parking_space_angle_weight = 15

        self.end_line_angle_weight = -50

        self.out_line_position_weight = 0.5
        self.out_line_angle_weight = 10
        
        self.obstacle_in_right = False          # 최종 결과 (update_searching에서 쓰는 값)
        self.obstacle_detect = 0           # 연속으로 조건 만족한 횟수
        self.required_num = 3         # 몇 번 연속일 때 True로 볼지
        
        self.parking_space_found_num = 0
        self.parking_space_required_num = 5


        self.parking_line_found = False
        self.parking_space_found = False
        self.end_line_found = False
        self.out_line_found = False

        self.end_line_stop_y = 1250
        self.end_line_stering = 1000

        # Parking State
        self.state = "SEARCHING_SPACE"
        # self.state = "ALIGN_TO_SPACE"

        self.enter_stopped_time = None
        
        # Subscribers
        self.create_subscription(LaserScan, "/scan_raw", self.lidar_callback, 10)
        self.create_subscription(ParkingLot, "/front_parking_line", self.parking_line_callback, 10)
        self.create_subscription(OutLine, "/front_out_line", self.out_line_callback, 10)
        self.create_subscription(ParkingSpace, "/rear_parking_space", self.parking_space_callback, 10)
        self.create_subscription(EndLine, "/rear_end_line", self.end_line_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(MotionCommand, "/motion_command", 10)
        self.state_pub = self.create_publisher(State, "/motion_state", 10)


        # Timer 
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("ParkingNode initialized with MotionCommand(steering, left_speed, right_speed).")
    

    # -------------------------
    # Callbacks
    # -------------------------
    def lidar_callback(self, msg: LaserScan):
        # SEARCHING 상태에서만 장애물 체크
        if self.state == "SEARCHING_SPACE":
            # 우측 감지
            condition_met = self.check_obstacle_sector(msg, target_deg=270.0)

            if condition_met:
                self.obstacle_detect += 1
                if self.obstacle_detect >= self.required_num:
                    self.obstacle_in_right = True
            else:
                self.obstacle_detect = 0

    def parking_line_callback(self, msg: ParkingLot):
        self.parking_line_found = msg.found
        if self.parking_line_found:
            self.parking_line_x = msg.x
            self.parking_line_yaw = msg.yaw

    def out_line_callback(self, msg: OutLine):
        self.out_line_found = msg.found
        if self.out_line_found:
            self.out_line_x = msg.x
            self.out_line_y = msg.y
            self.out_line_yaw = msg.yaw

    def parking_space_callback(self, msg: ParkingSpace):
        self.parking_space_found = msg.found
        if self.parking_space_found:
            self.parking_space_x = msg.x
            self.parking_space_yaw = msg.yaw
    
    def end_line_callback(self, msg: EndLine):
        self.end_line_found = msg.found
        if self.end_line_found:
            self.end_line_x = msg.x
            self.end_line_y = msg.y
            self.end_line_yaw = msg.yaw

    # -------------------------
    # Main control loop
    # -------------------------
    def control_loop(self):
        self.publish_state()
        if self.state == "SEARCHING_SPACE":
            # self.get_logger().info("🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴")
            self.update_searching()

        elif self.state == "ALIGN_TO_SPACE":
            # self.get_logger().info("🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠")
            self.update_alignment()

        elif self.state == "PARKING_IN":
            # self.get_logger().info("🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡")
            self.update_parking()

        elif self.state == "STOPPED":
            self.get_logger().info("🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢")
            self.update_stopped()

        elif self.state == "ESCAPE":
            self.get_logger().info("🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵")
            self.update_escape()
        
        elif self.state == "TURN_RIGHT":
            # self.get_logger().info("🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣")
            self.turn_right()

    # -------------------------
    # State behaviors
    # -------------------------
    def update_searching(self):
        """1) 공간 찾기 전: 직진"""
        if self.parking_line_found:
            steer_position = (self.parking_line_x- 0.7) * self.parking_line_position_weight
            if self.parking_line_yaw < 0:
                yaw = 10 / 0.57 * (self.parking_line_yaw + 1.57)
            else: 
                yaw = 10 / 0.57 * (self.parking_line_yaw - 1.57)

            steer_angle = yaw * self.parking_line_angle_weight
            
            mapped_steering = steer_position + steer_angle
            mapped_speed = 100
            
            # self.get_logger().info(f"\nposition: {steer_position:.1f}\nangle: {steer_angle:.1f}\n최종 steer: {mapped_steering:.1f}\n\n\n")
            
            self.publish_cmd(steering=int(mapped_steering), speed=int(mapped_speed))
        
        if self.obstacle_in_right:
            self.get_logger().info("Parking space found → ALIGN_TO_SPACE")
            self.state = "ALIGN_TO_SPACE"
            return
        
    def update_alignment(self):
        """2) 주차공간 충분히 잡힐 때까지 좌회전"""
        self.publish_cmd(steering=-10, speed=100)

        if self.parking_space_found:
            self.parking_space_found_num += 1
            if self.parking_space_found_num > self.parking_space_required_num:
                self.state = "PARKING_IN"
        else:
            self.parking_space_found_num = 0

    def update_parking(self):
        """3) 후진하여 주차"""
        
        if self.end_line_found:
            self.get_logger().info(f"\nend line y: {self.end_line_y:.1f}\nend line yaw: {self.end_line_yaw:.3f}\n")
            if self.end_line_y >= self.end_line_stering:

                mapped_steering = self.end_line_yaw * self.end_line_angle_weight
                mapped_steering = max(min(mapped_steering, 10.0), -10.0)
                
                mapped_speed = -100
                self.get_logger().info(f"\n최종 steer: {mapped_steering:.1f}\n\n\n")
                self.publish_cmd(steering=int(mapped_steering), speed=int(mapped_speed))

                if self.end_line_y >= self.end_line_stop_y:
                    self.state = "STOPPED"
                    return
                return
            
        elif not self.parking_space_found: # 인식 못하면 다시 좌회전 전진
            self.publish_cmd(steering=-10, speed=100)
            return
        
        steer_position = (self.parking_space_x - 0.5) * self.parking_space_position_weight
        steer_angle = self.parking_space_yaw * self.parking_space_angle_weight

        steer_position = -steer_position 
        yaw_target = math.pi / 2.0
        yaw_error = self.parking_space_yaw - yaw_target

        # 각도 wrap (-pi ~ pi)
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        steer_angle = yaw_error * self.parking_space_angle_weight
        steer_angle = -steer_angle  # 후진 기준 보정

        # --- 3) Steering 합성 ---
        mapped_steering = steer_position + steer_angle
        mapped_steering = max(min(mapped_steering, 10.0), -10.0)
        self.get_logger().info(f"\nposition: {steer_position:.1f}\nangle: {steer_angle:.1f}\n최종 steer: {mapped_steering:.1f}\n\n\n")

        mapped_speed = -100
        
        self.publish_cmd(steering=int(mapped_steering), speed=int(mapped_speed))


    def update_stopped(self):
        """4) 최종 정지"""
        self.publish_cmd(steering=0, speed=0)
        
        # 상태 진입 시간이 없다면 지금 시간 기록
        if self.enter_stopped_time is None:
            self.enter_stopped_time = self.get_clock().now()
            return

        # 경과시간 계산
        elapsed = (self.get_clock().now() - self.enter_stopped_time).nanoseconds / 1e9

        # 3초 지났는지 확인
        if elapsed >= 3.0:
            self.get_logger().info("정지 3초 완료.")
            self.enter_stopped_time = self.get_clock().now()
            self.state = "ESCAPE"
        else:
            self.get_logger().info(f"정지 유지 중... {elapsed:.1f}s / 3.0s")

    def update_escape(self):
        """5) 주차선 밟지 않게 직진"""
        self.publish_cmd(steering=0, speed=100)
        
        # 경과시간 계산
        elapsed = (self.get_clock().now() - self.enter_stopped_time).nanoseconds / 1e9

        # 3초 지났는지 확인
        if elapsed >= 3.0:
            self.get_logger().info("전진 3초 완료.")
            self.enter_stopped_time = None
            self.state = "TURN_RIGHT"
        else:
            self.get_logger().info(f"전진 유지 중... {elapsed:.1f}s / 3.0s")

    def turn_right(self):
        """6) out_line 찾아 종료"""
        if self.out_line_found:
            steer_position = (self.out_line_x - 0.5) * self.out_line_position_weight
            steer_angle = self.out_line_yaw * self.out_line_angle_weight
            
            mapped_steering = steer_position + steer_angle
            mapped_steering = max(min(mapped_steering, 10.0), -10.0)
            mapped_speed = 100
            
            self.get_logger().info(f"position: {steer_position:.1f}\nangle: {steer_angle:.1f}\n최종 steer: {mapped_steering:.1f}\n\n\n")
            self.publish_cmd(steering=int(mapped_steering), speed=int(mapped_speed))
        
        else:
            self.publish_cmd(steering=10, speed=100)

    # -------------------------
    # Lidar-obstacle
    # -------------------------
    def check_obstacle_sector(self, msg: LaserScan, target_deg: float) -> bool:
        """라이다 특정 각도(target_deg ± margin) 구간에서 장애물을 검사하는 함수"""

        min_deg = target_deg - self.margin
        max_deg = target_deg + self.margin

        total_in_window = 0
        obstacle_count = 0

        angle = msg.angle_min  # 시작각 (rad)

        for r in msg.ranges:
            angle_deg = math.degrees(angle)

            # target_deg ± margin 범위만 검사
            if min_deg <= angle_deg <= max_deg:
                total_in_window += 1

                # 유효 + 3m 이내면 장애물
                if (not math.isinf(r)) and (not math.isnan(r)) and r < 3.0:
                    obstacle_count += 1

            angle += msg.angle_increment

        # 절반 이상 감지 시 True
        if total_in_window > 0:
            ratio = obstacle_count / total_in_window
            return ratio >= 0.5

        return False



    # -------------------------
    # MotionCommand 생성
    # -------------------------
    def publish_cmd(self, steering: int, speed: int):
        """
        steering: int32
        speed: int32  → left_speed = speed, right_speed = speed
        """
        msg = MotionCommand()
        msg.steering = int(steering)
        msg.left_speed = int(speed)
        msg.right_speed = int(speed)

        self.cmd_pub.publish(msg)

    # -------------------------
    # Current State 생성
    # -------------------------
    def publish_state(self):
        msg = State()
        msg.state = self.state
        self.state_pub.publish(msg)    

def main(args=None):
    rclpy.init(args=args)
    node = ParkingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()