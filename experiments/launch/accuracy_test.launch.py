#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription
import os
from ament_index_python.packages import get_package_share_directory
from datetime import datetime

# select the cameras to be used


cameras = {"realsense", "oakd"}


def generate_launch_description():

    # Define the output directory for the rosbag recordings with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rosbag_output_dir = os.path.join(current_dir, '..', '..', '..', '..', '..', 'src', 'rwr_system', 'src', 'ingress', 'paper_experiments', 'rosbag_recordings', f'recording_{timestamp}')

    urdf = os.path.join(
    get_package_share_directory('viz'),
    "models",
    "orca_v1",
    "urdf",
    "orca_v1.urdf")

    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    

    return LaunchDescription(
        [
          
            IncludeLaunchDescription(
            launch_description_source=os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch', 'rs_launch.py'
                ),
            ),

            Node(
                package='ingress',
                executable='align_cam2april_node.py',
                name='align_cam2april_node',
                output='screen',
                ),

    
            # # ACCURACY TEST NODE
            # Node(
            #     package="ingress",
            #     executable="accuracy_node.py",
            #     name="accuracy_node",
            #     output="log",
            #     parameters=[
            #         {"motion_duration": 4.0},
            #         {"recalibration_interval": 10.0},
            #         {"flexion_scalar": 0.4},
            #         {"signal_type": "sine"},  # Use "step" if you want step signals, "sine" for sign waves
            #         {"retarget/hand_scheme": os.path.join(
            #             get_package_share_directory("viz"),
            #             "models",
            #             "orca_v1",
            #             "scheme_orca_v1.yaml",)
            #         },
            #     ],
            # ),
            
            
            # # VISUALIZATION NODE
            # Node(
            #     package="viz",
            #     executable="visualize_joints.py",
            #     name="visualize_joints",
            #     parameters=[
            #         {
            #             "scheme_path": os.path.join(
            #                 get_package_share_directory("viz"),
            #                 "models",
            #                 "orca_v1",
            #                 "scheme_orca_v1.yaml",
            #             )
            #         }
            #     ],
            #     output="screen",
            # ),

            # Node(
            #     package='robot_state_publisher',
            #     executable='robot_state_publisher',
            #     name='robot_state_publisher',
            #     output='screen',
            #     parameters=[{'robot_description': robot_desc,}],
            #     arguments=[urdf]),
            
            # Node(
            #     package='rviz2',
            #     executable='rviz2',
            #     name='rviz2',
            #     output='screen', 
            #     arguments=['-d', os.path.join(get_package_share_directory('viz'), 'rviz', 'retarget_config_orca_v1.rviz')],
            #     ),

            # # Node to start recording OAK-D camera frames and commanded angles to a rosbag
            # ExecuteProcess(
            #     cmd=['ros2', 'bag', 'record', '--output', rosbag_output_dir, 'image_raw', '/hand/policy_output'],
            #     output='screen'
            # )
        ]
    )
