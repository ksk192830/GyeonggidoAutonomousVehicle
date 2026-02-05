from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        #################### CAMERA1(전방) ######################
        Node(
            package='camera_pkg',
            executable='image',
            name='cam0',
            namespace='cam0',
            parameters=[
                {'data_source': 'camera'},
                {'cam_num': 2},
                {'pub_topic': '/cam0/image_raw'},
                {'window_name': 'Raw 0'},
                {'show_image': False},
                {'timer_period': 0.5},
            ]
        ),

        #################### IMAGE SAVER (전방) ######################
        DeclareLaunchArgument(
            'cam0_save_dir',
            default_value='/home/kimmyungkyun/gyeonggi_ws/src/camera_pkg/camera_pkg/lib/parking_front_new',
            description='Directory to save images from cam0'
        ),
        DeclareLaunchArgument(
            'cam0_sub_topic',
            default_value='/cam0/image_raw',
            description='Image topic to subscribe for saving (cam0)'
        ),

        Node(
            package='camera_pkg',
            executable='image_saver',
            name='image_saver_cam0',
            namespace='cam0',
            output='screen',
            parameters=[
                {'save_dir': LaunchConfiguration('cam0_save_dir')},
                {'sub_topic': LaunchConfiguration('cam0_sub_topic')},
            ]
        ),

        #################### CAMERA2(후방) ######################
        Node(
            package='camera_pkg',
            executable='image',
            name='cam1',
            namespace='cam1',
            parameters=[
                {'data_source': 'camera'},
                {'rotate_mode': 1},
                {'cam_num': 4},
                {'pub_topic': '/cam1/image_raw'},
                {'window_name': 'Raw 1'},
                {'show_image': False},
                {'timer_period': 0.5},
            ]
        ),

        #################### IMAGE SAVER (후방) ######################
        DeclareLaunchArgument(
            'cam1_save_dir',
            default_value='/home/kimmyungkyun/gyeonggi_ws/src/camera_pkg/camera_pkg/lib/parking_rear_new',
            description='Directory to save images from cam1'
        ),
        DeclareLaunchArgument(
            'cam1_sub_topic',
            default_value='/cam1/image_raw',
            description='Image topic to subscribe for saving (cam1)'
        ),

        Node(
            package='camera_pkg',
            executable='image_saver',
            name='image_saver_cam1',
            namespace='cam1',
            output='screen',
            parameters=[
                {'save_dir': LaunchConfiguration('cam1_save_dir')},
                {'sub_topic': LaunchConfiguration('cam1_sub_topic')},
            ]
        ),
    ])
