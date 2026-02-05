import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from interfaces_pkg.msg import LaneInfo, MotionCommand
import re

class MotionNode(Node):
    def __init__(self):
        super().__init__('motion')

        # Subscriptions
        self.create_subscription(LaneInfo, '/cam0/lane_info', self.lane_info_callback, 10)
        self.create_subscription(String, 'traffic_light_result', self.traffic_callback, 10)

        # Publisher
        self.motion_pub = self.create_publisher(MotionCommand, 'motion_command', 10)

        # 신호등 감지 설정
        self.lane1_area_threshold =  9999999999#18000
        self.lane2_area_threshold =  9999999#15000  # lane2에서 1
        self.required_count = 3

        # 상태 변수
        self.is_changing_lane = False
        self.wait_for_traffic_clear = False
        self.current_lane = None
        self.target_lane = 2

        # 신호등 감지 카운터
        self.detect_counter = 0
        self.clear_counter = 0

        # lane 1 설정
        self.lane1_angle_weight = 0.75 # 원래 0.75
        self.lane1_position_weight = 0.05
        self.lane1_normal_speed =255
        self.lane1_lane_change_speed = 255

        # lane 2 설정
        self.lane2_angle_weight = 0.73 # 원래 0.7
        self.lane2_position_weight = 0.05
        self.lane2_normal_speed = 255
        self.lane2_lane_change_speed = 255

    def get_lane_config(self):
        if self.current_lane == 1:
            return (self.lane1_angle_weight,
                    self.lane1_position_weight,
                    self.lane1_normal_speed,
                    self.lane1_lane_change_speed)
        elif self.current_lane == 2:
            return (self.lane2_angle_weight,
                    self.lane2_position_weight,
                    self.lane2_normal_speed,
                    self.lane2_lane_change_speed)
        else:
            # 기본값 (차선 정보 없을 경우)
            return (0.6, 0.05, 255, 100)

    def traffic_callback(self, msg):
        match = re.match(r'Detected:\s*(\w+),\s*Area:\s*([\d.]+)(?:,\s*Color:\s*(\w+))?', msg.data)
        # self.get_logger().info(msg.data)
        if not match:
            return

        detected = match.group(1).lower() == 'true'
        area = float(match.group(2))
        # color = match.group(3)  # 색 정보 무시!
        

        # 현재 lane에 맞는 threshold 선택
        if self.current_lane == 1:
            threshold = self.lane1_area_threshold
        else :
            threshold = self.lane2_area_threshold

        # 신호등이 감지 됐으며 크기가 클 때
        if detected and area > threshold:
            self.clear_counter = 0

            # self.get_logger().info("🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦🚦")

            if self.wait_for_traffic_clear:
                return

            # 신호등 감지 시
            self.detect_counter += 1
            # self.get_logger().info(f"🔎 Traffic detected {self.detect_counter}")

            if self.detect_counter >= self.required_count:
                # 목표 차선 변경
                self.target_lane = 1 if self.target_lane == 2 else 2
                self.wait_for_traffic_clear = True
                
        
        # 신호등 미감지
        else:
            self.detect_counter = 0

            if self.wait_for_traffic_clear:
                self.clear_counter += 1
                # self.get_logger().info(f"🟤 Traffic not detected {self.clear_counter}")

                if self.clear_counter >= 100:
                    self.wait_for_traffic_clear = False
                    self.clear_counter = 0
                    
                    #신호등 다시 감지 시작
                    self.get_logger().info("🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢")

    def lane_info_callback(self, msg):
        self.current_lane = msg.lane_num
        steering_angle = msg.steering_angle
        vehicle_position_x = msg.vehicle_position_x
        self.get_logger().info(f"vehicle_position: {vehicle_position_x} ")

        # 차선 변경을 위한 상태 관리
        if self.current_lane != self.target_lane:
            self.is_changing_lane = False #True

        if self.is_changing_lane and self.current_lane == self.target_lane and abs(vehicle_position_x) <= 200:
            self.is_changing_lane = False
            # self.get_logger().info(f"✅ Lane change complete: now on lane {self.current_lane}, centered in lane")

        # 현재 차선 기준 설정값 가져오기
        angle_weight, position_weight, normal_speed, lane_change_speed = self.get_lane_config()

        cmd = MotionCommand()

        # 차선 변경 중일 때
        if self.is_changing_lane:
            steering_value = -7 if self.target_lane == 1 else 7
            cmd.left_speed = lane_change_speed
            cmd.right_speed = lane_change_speed
            self.get_logger().info(f"🔁")
        else:
            # 차선 변경이 완료되면 차선 유지
            mapped = (steering_angle / 50.0) * 10.0 * angle_weight
            adjust = - vehicle_position_x * position_weight
            
            adjust = max(-3, min(adjust, 3))
            steering_value = max(-10, min(mapped + adjust, 10))
            cmd.left_speed = normal_speed
            cmd.right_speed = normal_speed
            self.get_logger().info(f"mapped: {mapped:.1f}, adjust: {adjust:.1f}, 현재 steer: {steering_value:.1f}\n\n")


        cmd.steering = int(steering_value)
        self.motion_pub.publish(cmd)
        # self.get_logger().info(f"🎯 현재 차선: {self.current_lane}, 🎯목표a 차선: {self.target_lane}")

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
