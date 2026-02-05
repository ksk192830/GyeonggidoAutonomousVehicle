from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable

def generate_launch_description():

    # 각 카메라에 맞는 모델 경로
    cam0_model_path = "/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/model/parking_front.pt"
    cam1_model_path = "/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/model/parking_rear.pt"
    rviz_config = "/home/sg/gyeonggi_ws/src/control_pkg/rviz/final_parking.rviz"


    return LaunchDescription([
        #################### CAMERA1(전방) ######################
        Node(
            package='camera_pkg',
            executable='image',  # 실행할 노드 파일명
            name='cam0',
            namespace='cam0',
            parameters=[
                # {'data_source': 'image'},
                {'data_source': 'camera'},  # camera, video, image 선택

                {'cam_num': 2},
                {'img_dir': '/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/lib/Jjin_track/parking_front'},
                {'pub_topic': '/cam0/image_raw'},
                {'window_name': 'Raw 0'},
                {'show_image': False},

                {'timer_period': 0.1},
            ]
        ),

        #################### CAMERA2(후방) ######################
        Node(
            package='camera_pkg',
            executable='image',  # 실행할 노드 파일명
            name='cam1',
            namespace='cam1',
            parameters=[
                # {'data_source': 'image'},
                {'data_source': 'camera'},  # camera, video, image 선택
                {'rotate_mode': 1},

                {'cam_num': 4},
                {'img_dir': '/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/lib/Jjin_track/parking_rear'},
                {'pub_topic': '/cam1/image_raw'},
                {'window_name': 'Raw 1'},
                {'show_image': False},
 
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
                # cpu or cuda:0
                {'device': 'cuda:0'},
                {'model_path': cam0_model_path},
                {'threshold': 0.5},

                # segmentation 수행할 state 목록
                {'allowed_states': ['SEARCHING_SPACE', 'TURN_RIGHT']}
            ],
            remappings=[
                ('image_raw', '/cam0/image_raw'),  
                ('detections', '/cam0/detections'),
                ('seg_vis', '/cam0/seg_vis')  
            ]
        ),

        #################### CAMERA1 detection ######################
        Node(
            package="camera_pkg",           
            executable="parking_front_detect",
            name="parking_front_detect",
            output="screen",
            parameters=[{
                "image_topic": "/cam0/image_raw",
                "detection_topic": "/cam0/detections",
                "viz_topic": "/front_viz",
            }],
        ),
        

        
        #################### CAMERA2 YOLO SEG ######################
        SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', '0'),
        Node(
            package='camera_pkg',
            executable='yolo_seg',  
            name='yolo_seg1',
            namespace='cam1',
            output='screen',
            parameters=[
                # cpu or cuda:0
                {'device': 'cuda:0'},
                {'model_path': cam1_model_path},
                {'threshold': 0.5},

                # segmentation 수행할 state 목록
                {'allowed_states': ['ALIGN_TO_SPACE', 'PARKING_IN']}
            ],
            remappings=[
                ('image_raw', '/cam1/image_raw'),  
                ('detections', '/cam1/detections'),
                ('seg_vis', '/cam1/seg_vis')  
            ]
        ),

        #################### CAMERA2 detection ######################
        Node(
            package="camera_pkg",           
            executable="parking_rear_detect",
            name="parking_rear_detect",
            output="screen",
            parameters=[{
                "image_topic": "/cam1/image_raw",
                "detection_topic": "/cam1/detections",
                "viz_topic": "/rear_viz",
            }],
        ),

        # #################### PARKING MAP VIZ ######################
        # Node(
        #     package="decision_making_pkg",
        #     executable="parking_map_viz",   # ← 노드 파일명
        #     name="parking_map_viz",
        #     output="screen",
        #     parameters=[{
        #         "yaml_path": "/home/sg/gyeonggi_ws/src/decision_making_pkg/decision_making_pkg/config/parking_map.yaml",
        #         "frame_id": "map",
        #         "topic": "/parking_map/markers",

        #         "show_parking_lot": True,
        #         "selected_slot_id": "slot_02",   # "" 이면 전체 표시
        #     }]
        # ),

        #################### LIDAR SCAN ######################
        Node(
            package='lidar_pkg',              # 라이다 패키지 이름
            executable='scan',          # setup.cfg에 등록한 이름 사용!
            name='lidar_scan',
            output='screen',
            parameters=[
                {'pub_topic': '/scan_raw'},    # MotionNode에서 구독하는 토픽 이름과 맞추기
                {'lidar_port': '/dev/ttyUSB0'},
                {'allowed_states': ['SEARCHING_SPACE']},
            ]
        ),
        # #################### LIDAR Cluster ######################
        # Node(
        #     package='lidar_pkg',          # your package name
        #     executable='scan_cluster',        # entry in setup.py console_scripts
        #     name='scan_cluster',
        #     output='screen',
        #     parameters=[
        #         {
        #             'sub_topic': 'scan_raw',
        #             'window_name': 'Lidar Scan Viewer',
        #             'enable_viz': True    # or False to disable OpenCV
        #         }
        #     ]
        # ),

        #################### 주차 로직 ######################
         
        Node(
            package='decision_making_pkg',
            executable='motion_parking',
            name='motion_parking',
            output='screen'
        ),

        #################### CONTROL ######################
        # Node(
        #     package='control_pkg',  
        #     executable='parking_control',  
        #     name='control_node',
        #     output='screen'
        # ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),

        
    ])
