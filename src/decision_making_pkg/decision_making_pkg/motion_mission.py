#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from interfaces_pkg.msg import LaneInfo, MotionCommand
from sensor_msgs.msg import LaserScan
import re
import math


class MotionNode(Node):
    def __init__(self):
        super().__init__('motion_mission')


        # Subscriptions
        self.create_subscription(LaneInfo, '/cam0/lane_info', self.lane_info_callback, 10)
        self.create_subscription(String, 'traffic_light_result', self.traffic_callback, 10)
        self.create_subscription(String, 'obstacle_result', self.obstacle_callback, 10)
        self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, 10)

        # Publisher
        self.motion_pub = self.create_publisher(MotionCommand, 'motion_command', 10)

        # 1차선 물체 크기별 구간 임계값 (튜닝)
        # self.far_height  = 300  # m 단위, 이보다 멀면 1. 정지
        # self.accurate_height = 450   # 2. 직진
        # self.ambiguous_height = 480  # 3. 정지

        # traffic light
        self.traffic_area_threshold = 28000
        self.red_required_count = 3
        self.red_clear_required_count = 3

        # 장애물 감지
        self.lidar_obstacle_counter = 0
        self.lidar_obstacle_threshold = 3
        self.obstacle_flag = False


        # 상태 변수
        self.current_lane = 2
        self.target_lane = 2
        self.is_changing_lane = False
        self.wait_for_red_clear = False
        self.once = True

        self.backward_motion = False
        self.forward_motion = False

        # LiDAR state
        self.latest_lidar_avg = None
        self.prev_lidar_in_range = False

        # 한 번만 실행됐는지 표시할 플래그
        self.lidar_lane_change_executed    = False

        # 라이다로 2차선에서 1차선 변경용
        self.lidar_lane_change_counter   = 0
        self.lidar_lane_change_threshold = 5

        # 연속 장애물 없음 감지용 변수
        self.no_obstacle_counter = 0
        self.no_obstacle_threshold = 3

        # lane 1 설정
        self.lane1_angle_weight = 0.75
        self.lane1_position_weight = 0.05
        self.lane1_normal_speed = 200
        self.lane1_lane_change_speed = 200

        # lane 2 설정
        self.lane2_angle_weight = 0.7
        self.lane2_position_weight = 0.05
        self.lane2_normal_speed = 200
        self.lane2_lane_change_speed = 200


    def lidar_callback(self, msg: LaserScan):
        if self.is_changing_lane:
            return
        
        # 차선에 따라 볼 구간 설정
        if self.target_lane == 1:
            start_deg, end_deg = 267.0, 270.0
              
        else: 
            start_deg, end_deg = 90, 93

        # deg → rad 변환
        start_angle = start_deg * math.pi / 180.0
        end_angle   = end_deg   * math.pi / 180.0

        # 인덱스 계산
        start_idx = int((start_angle - msg.angle_min) / msg.angle_increment)
        end_idx   = int((end_angle   - msg.angle_min) / msg.angle_increment)

        # 배열 경계(clamp) 처리
        start_idx = max(0, min(start_idx, len(msg.ranges)-1))
        end_idx   = max(0, min(end_idx,   len(msg.ranges)))

        # 범위 슬라이스 후 유효한 값만 필터링
        sector   = msg.ranges[start_idx:end_idx]
        filtered = [r for r in sector if 0.0 < r < float('inf')]

        # 평균 거리 계산
        if filtered:
            self.latest_lidar_avg = sum(filtered) / len(filtered)
        else:
            self.latest_lidar_avg = None

        # # # ──────── 장애물 유무 로그 출력 ────────
        # if self.latest_lidar_avg is None:
        #     self.get_logger().info("[LiDAR] 유효한 거리 값이 없습니다")
        # elif self.latest_lidar_avg < 1.0:
        #     self.get_logger().info(f"[LiDAR] 🚧 장애물 감지! 평균 거리 = {self.latest_lidar_avg:.2f} m")
        # else:
        #     self.get_logger().info(f"[LiDAR] ✅ 장애물 없음, 평균 거리 = {self.latest_lidar_avg:.2f} m")


    def obstacle_callback(self, msg: String):
        if self.is_changing_lane:
            return
        
        # 카메라 기반 장애물 메시지 파싱
        match = re.match(r'Detected:\s*(\w+),\s*Area:\s*([\d.]+),\s*Height:\s*([\d.]+)', msg.data)

        if not match:
            return
        self.obstacle_detected = (match.group(1).lower() == 'true')
        # area = float(match.group(2))
        self.obstacle_height = float(match.group(3))
        # self.get_logger().info(f"[Obstacle] detected={obstacle_detected}, area={area}, lane={self.current_lane}")


    def get_lane_config(self):  
        if self.current_lane == 1:
            return (self.lane1_angle_weight,
                    self.lane1_position_weight,
                    self.lane1_normal_speed,
                    self.lane1_lane_change_speed)
        else :
            return (self.lane2_angle_weight,
                    self.lane2_position_weight,
                    self.lane2_normal_speed,
                    self.lane2_lane_change_speed)
    
    def traffic_callback(self, msg):
        match = re.match(r'Detected:\s*(\w+),\s*Area:\s*([\d.]+)(?:,\s*Color:\s*(\w+))?', msg.data)
        # self.get_logger().info(msg.data)
        if not match:
            return
        
        self.traffic_light_detected = match.group(1).lower() == 'true'
        self.traffic_light_area = float(match.group(2))
        self.traffic_light_color = (match.group(3) or '').lower()

    def lane_info_callback(self, msg: LaneInfo):
        

        # Update for obstacle steering calculation
        self.current_lane = msg.lane_num
        steering_angle = msg.steering_angle
        vehicle_position_x = msg.vehicle_position_x
        
        angle_weight, position_weight,normal_speed, lane_change_speed =  self.get_lane_config()
        
        cmd = MotionCommand()
        mapped = (steering_angle / 50.0) * 10.0 * angle_weight
        adjust = - vehicle_position_x * position_weight
        adjust = max(-3, min(adjust, 3))
        steering_value = max(-10, min(mapped + adjust, 10))


        # ================= 목표를 2차선에서  1차선 변경 ================
        if self.current_lane == 2 and not self.lidar_lane_change_executed:
            if self.latest_lidar_avg is not None and self.latest_lidar_avg < 1.0:
                self.lidar_lane_change_counter += 1
            else:
                self.lidar_lane_change_counter = 0

            if self.lidar_lane_change_counter >= self.lidar_lane_change_threshold:
                if self.once:
                    self.target_lane = 1
                    self.once = False
                self.is_changing_lane = True
                self.lidar_lane_change_executed = True    # **한 번 실행 처리**
                # self.get_logger().info(
                #     f"🚧 LiDAR 장애물 {self.lidar_lane_change_threshold}회 연속 감지 "
                #     f"🔄 🔄 🔄 🔄 🔄 🔄 🔄 (avg={self.latest_lidar_avg:.2f}m) → 1차선 변경"
                # )
                self.lidar_lane_change_counter = 0
                return
        # ==========================================================

        # ================= 목표를 1차선에서  2차선 변경 ================
        if self.current_lane == 1 :
            if self.latest_lidar_avg is not None and self.latest_lidar_avg < 1.0:
                self.lidar_lane_change_counter += 1
            else:
                self.lidar_lane_change_counter = 0

            if self.lidar_lane_change_counter >= self.lidar_lane_change_threshold:
                self.target_lane = 2
                self.is_changing_lane = True
                self.get_logger().info(
                    f"🚧 LiDAR 장애물 {self.lidar_lane_change_threshold}회 연속 감지 "
                    f"🔄 🔄 🔄 🔄 🔄 🔄 🔄 (avg={self.latest_lidar_avg:.2f}m) → 1차선 변경"
                )
                self.lidar_lane_change_counter = 0
                return
        # ==========================================================
        if self.target_lane != self.current_lane :
            self.is_changing_lane = True
             

        # Lane change logic
        if self.is_changing_lane:
            # self.get_logger().info(f"🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄")
            cmd.steering = -10 if self.target_lane == 1 else 10
            cmd.left_speed = lane_change_speed
            cmd.right_speed = lane_change_speed


            # Lane change 상태 해제
            if self.current_lane == self.target_lane and abs(vehicle_position_x) <= 200:
                self.is_changing_lane = False
                self.obstacle_flag = False
                self.latest_lidar_avg = None
                self.get_logger().info(f"✅ Lane change complete: now on lane {self.current_lane}")
        

        # ==================================== 신호등 정지 로직 ====================================
        elif self.traffic_light_detected and self.traffic_light_area > self.traffic_area_threshold and self.traffic_light_color == 'red':
            cmd.left_speed = 0
            cmd.right_speed = 0
            cmd.steering = 0
            self.get_logger().info("🛑 Red light detected: stopping vehicle")

        elif self.traffic_light_detected:
            cmd.left_speed = normal_speed
            cmd.right_speed = normal_speed
            cmd.steering = int(steering_value)
            # self.red_detect_counter = 0
            # if self.wait_for_red_clear:
            #     self.red_clear_counter += 1
            #     if self.red_clear_counter >= self.red_clear_required_count:
            #         self.wait_for_red_clear = False
            #         self.red_clear_counter = 0
            #         self.get_logger().info("🟢 Red light cleared: resuming")
        # ================================================================================================  

    
        else:     
            cmd.left_speed = normal_speed
            cmd.right_speed = normal_speed
            cmd.steering = int(steering_value)
            # self.get_logger().info(f"mapped: {mapped:.1f}, adjust: {adjust:.1f}, 현재 steer: {steering_value:.1f}")

        self.motion_pub.publish(cmd)
    
def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('MotionNode interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
