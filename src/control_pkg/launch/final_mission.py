import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable
import os

def generate_launch_description():
     # 각 카메라에 맞는 모델 경로
    cam0_model_path = "/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/model/track.pt"
    cam0_model_path_cw = "/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/model/crosswalk.pt"
    rviz_config = "/home/sg/gyeonggi_ws/src/control_pkg/rviz/final_mission.rviz"

    return LaunchDescription([
        
        #################### CAMERA1(LANE) ######################
        
        Node(
            package='camera_pkg',
            executable='image',  # 실행할 노드 파일명
            name='cam0',
            namespace='cam0',
            parameters=[
                {'data_source': 'image'},  # camera, video, image 선택
                # {'data_source': 'camera'},

                {'cam_num': 0}, # 포트 확인   ls /dev/video*
                {'img_dir': '/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/lib/Jjin_track/mission'},
                {'pub_topic': '/cam0/image_raw'},
                {'window_name': 'Raw 0'},
                {'show_image': False},
                # {'timer_period': 0.03},  # float형으로, fps조절 
                {'timer_period': 0.1},
            ]
        ),
        
        #################### CAMERA1 YOLO SEG ######################
        SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', '0'),
        Node(
            package='camera_pkg',
            executable='yolo_seg',  
            name='yolo_seg0',
            namespace='cam0',
            output='screen',
            parameters=[
                {'device': 'cuda:0'},
                # {'device': 'cpu'},
                {'model_path': cam0_model_path},
                {'threshold': 0.5}
            ],
            remappings=[
                ('image_raw', '/cam0/image_raw'),  # 카메라의 image_raw 토픽과 매핑
                ('detections', '/cam0/detections'),  # YOLO가 감지한 결과를 detections에 퍼블리시
                ('seg_vis', '/cam0/seg_vis')
            ]
        ),

        #################### CAMERA1 YOLO SEG_CW ######################
        SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', '0'),
        Node(
            package='camera_pkg',
            executable='yolo_seg',  
            name='yolo_seg1',
            namespace='cam0',
            output='screen',
            parameters=[
                {'device': 'cuda:0'},
                # {'device': 'cpu'},
                {'model_path': cam0_model_path_cw},
                {'threshold': 0.5}
            ],
            remappings=[
                ('image_raw', '/cam0/image_raw'),  # 카메라의 image_raw 토픽과 매핑
                ('detections', '/cam0/detections_cw'),  # YOLO가 감지한 결과를 detections에 퍼블리시
                ('seg_vis', '/cam0/seg_vis_cw')
            ]
        ),

        #################### LANE DETECT ######################
        Node(
            package='camera_pkg',
            executable='lane',  
            name='lane_detector_cam0',
            namespace='cam0',
            parameters=[
                {'camera_topic': '/cam0/image_raw'},
                {'detection_topic': '/cam0/detections'},
                ('seg_vis', '/cam0/seg_vis')
            ],
            output='screen'
        ),

        #################### TRAFFIC LIGHT DETECT ######################
        Node(
            package='camera_pkg',
            executable='traffic_light',
            name='traffic_light',
            output='screen',
            parameters=[]
        ),

        #################### OBSTACLE DETECT ######################
        Node(
            package='camera_pkg',
            executable='obstacle',
            name='obstacle',
            output='screen',
            parameters=[]
        ),

        #################### CROSS WALK DETECT ######################
        Node(
            package='camera_pkg',
            executable='cross_walk_detect',  
            name='cross_walk',
            output='screen',
            parameters=[],
        ),
        
        #################### LIDAR SCAN ######################
        Node(
            package='lidar_pkg',              # 라이다 패키지 이름
            executable='scan',          
            name='lidar_scan',
            output='screen',
            parameters=[
                {'pub_topic': '/scan_raw'},    # MotionNode에서 구독하는 토픽 이름과 맞추기
                {'lidar_port': '/dev/ttyUSB0'},
            ]
        ),

        # #################### LIDAR VISUALIZER ######################
        # Node(
        #     package='lidar_pkg',
        #     executable='scan_viz',   
        #     name='scan_viz',
        #     output='screen',
        #     parameters=[
        #         {'sub_topic': '/scan_raw'},
        #         {'window_name': 'Lidar Scan Viewer'},
        #     ]
        # ),

        
        #################### MOTION ######################
        Node(
            package='decision_making_pkg',  
            executable='motion_mission',  
            name='motion_mission',
            output='screen'
        ),
        
        #################### CONTROL ######################
        # Node(
        #     package='control_pkg',  
        #     executable='control',  
        #     name='control_node',
        #     output='screen'
        # )

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),
        
    ])

