#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from interfaces_pkg.msg import ParkingSpace, MotionCommand


import math

class ParkingNode(Node):

    def __init__(self):
        super().__init__("motion_parking")

        # Parking State
        self.state = "SEARCHING_SPACE"

        # 후진 시 yaw 가중치
        self.yaw_weight = 0.5
        self.direction_weight = 0.7
        self.last_weight = 3
        
        # 라이다 스캔 ± 범위
        self.margin = 4.0

        # 장애물 감지
        self.obstacle_in_right = False          # 최종 결과 (update_searching에서 쓰는 값)
        self.obstacle_detect = 0           # 연속으로 조건 만족한 횟수
        self.required_num = 3         # 몇 번 연속일 때 True로 볼지
        self.space_detect = 0
        self.back_clear = 0

        # State
        self.parking_complete = False
        self.in_parking_space = False
        self.enter_stopped_time = None

        # ParkingSpace msg data
        self.parking_found = False
        self.parking_x = 0.0
        self.parking_y = 0.0
        self.parking_yaw = 0.0
        
        # Subscribers
        self.create_subscription(LaserScan, "/scan_raw", self.lidar_callback, 10)
        self.create_subscription(ParkingSpace, "/parking_space", self.parking_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(MotionCommand, "/motion_command", 10)

        # Timer (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

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
                
        elif self.state == "PARKING_IN":
            if not self.in_parking_space:
                # 우측 감지
                condition_met1 = self.check_obstacle_sector(msg, target_deg=270.0)
                # 좌측 감지
                condition_met2 = self.check_obstacle_sector(msg, target_deg=90.0)
                
                if condition_met1 and condition_met2:
                    self.space_detect += 1
                    if self.space_detect >= self.required_num:
                        self.in_parking_space = True
                        self.get_logger().info("✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
                else:
                    self.space_detect = 0
                # self.get_logger().info("주차자리로 가는 중")
            
            else:
                # 우측 감지
                condition_met1 = self.check_obstacle_sector(msg, target_deg=270.0)
                # 좌측 감지
                condition_met2 = self.check_obstacle_sector(msg, target_deg=90.0)
                
                if not condition_met1 and not condition_met2:
                    self.back_clear += 1
                    if self.back_clear >= self.required_num:
                        self.parking_complete= True
                        self.get_logger().info("🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯")
                else:
                    self.back_clear = 0
                # self.get_logger().info("마무리 후진 중")

                


            
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

    def parking_callback(self, msg: ParkingSpace):
        self.parking_found = msg.found
        self.parking_x = msg.x
        self.parking_y = msg.y
        self.parking_yaw = msg.yaw

    # -------------------------
    # Main control loop
    # -------------------------
    def control_loop(self):
        if self.state == "SEARCHING_SPACE":
            self.update_searching()

        elif self.state == "ALIGN_TO_SPACE":
            self.update_alignment()

        elif self.state == "PARKING_IN":
            self.update_parking()

        elif self.state == "STOPPED":
            self.update_stopped()

        elif self.state == "ESCAPE":
            self.update_escape()
        
        elif self.state == "TURN_RIGHT":
            self.turn_right()

    # -------------------------
    # State behaviors
    # -------------------------
    def update_searching(self):
        """1) 공간 찾기 전: 직진"""
        self.publish_cmd(steering=0, speed=100)
        if self.obstacle_in_right:
            self.get_logger().info("Parking space found → ALIGN_TO_SPACE")
            self.state = "ALIGN_TO_SPACE"
            # self.state = "ESCAPE"
            return

  

    def update_alignment(self):
        """2) 주차공간의 yaw 에 맞게 좌회전"""
        self.publish_cmd(steering=-10, speed=100) 
        # -> 0 / ^ 90 / <- 180
        if not self.parking_found:
            return
        yaw = self.parking_yaw
        self.get_logger().info(f'yaw: {yaw}\n\n\n')
        distance = math.hypot(self.parking_x, self.parking_y)

        # 0 ~ 30 or 150 ~ 180 될 때까지 좌회전
        if (yaw < 50 or yaw > 130) and distance > 1.3:
            self.get_logger().info("Alignment complete → PARKING_IN")
            self.state = "PARKING_IN"
            return


    def update_parking(self):
        """3) 후진하여 주차"""
        if not self.parking_found:
            return
        
        yaw = self.parking_yaw
        direction_yaw = math.degrees(math.atan2(self.parking_y , self.parking_x))
        distance = math.hypot(self.parking_x, self.parking_y)
        # self.get_logger().info(f'👉👉 distance: {distance:.1f}\n\n\n')
        
        # self.get_logger().info(f'👉👉 yaw: {yaw:.1f}\n\n\n')
        if yaw < 90:
            steer_yaw = yaw * self.yaw_weight 
        else:
            yaw = yaw - 180 
            steer_yaw = yaw * self.yaw_weight 
        # self.get_logger().info(f'👉👉 steer_yaw: {steer_yaw:.1f}\n\n\n')
        if direction_yaw < 0:
            direction = (direction_yaw + 90) * self.direction_weight
        else:  
            direction = (direction_yaw - 90) * self.direction_weight
        
        if distance < 0.2 : 
            direction = 0
        elif distance < 0.5 : 
            direction = max(-1, min(direction, 1))

        steer = max(-10, min(int(steer_yaw + direction), 10))

        # self.get_logger().info(f'👉 steer_yaw: {steer_yaw:.1f} 👉 direction: {direction:.1f} \n👉 steer: {steer:.1f}\n\n\n')
        self.publish_cmd(steering=steer, speed=-100)
        self.get_logger().info("Parking in...")
        
        if self.parking_complete :
            self.enter_stopped_time = self.get_clock().now()
            self.state = "STOPPED"

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
        # else:
        #     # self.get_logger().info(f"정지 유지 중... {elapsed:.1f}s / 3.0s")

    def update_escape(self):
        self.publish_cmd(steering=0, speed=100)
        
        # 경과시간 계산
        elapsed = (self.get_clock().now() - self.enter_stopped_time).nanoseconds / 1e9

        # 3초 지났는지 확인
        if elapsed >= 1.0:
            self.get_logger().info("전진 3초 완료.")
            self.enter_stopped_time = None
            self.state = "TURN_RIGHT"
        # else:
        #     # self.get_logger().info(f"전진 유지 중... {elapsed:.1f}s / 3.0s")

    def turn_right(self):
        if not self.parking_found:
            self.publish_cmd(steering=3, speed=100)
            return
        
        yaw = self.parking_yaw
        if 85 < yaw < 95:
            yaw = yaw - 90
            steer = yaw * self.last_weight

        else: 
            yaw = abs(yaw - 90)
            steer = yaw * self.last_weight
        self.publish_cmd(steering=steer, speed=100)
        # self.get_logger().info(f'👉 steer: {steer:.1f} \n\n')
        

    # -------------------------
    # Helper: MotionCommand 생성
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

def main(args=None):
    rclpy.init(args=args)
    node = ParkingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
