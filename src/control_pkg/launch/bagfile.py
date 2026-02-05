from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from datetime import datetime

def generate_launch_description():
    # Get current time for unique bag name
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    default_bag_name = f"rosbag_record_{timestamp}"

    bag_name_arg = DeclareLaunchArgument(
        'bag_name',
        default_value=default_bag_name,
        description='Name of the bag file (directory)'
    )

    bag_name = LaunchConfiguration('bag_name')

    # Topics to record
    topics = [
        '/scan_raw',
        '/cam0/image_raw',
        '/cam1/image_raw'
    ]

    record_process = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', bag_name] + topics,
        output='screen'
    )

    return LaunchDescription([
        bag_name_arg,
        record_process
    ])