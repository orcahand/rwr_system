from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, Shutdown
import os
from ament_index_python.packages import get_package_share_directory

# select the cameras to be used

cameras = {"front_view": True, "side_view": True, "wrist_view": True}


def generate_launch_description():
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
            # HAND CONTROLLER NODE
            Node(
                package="hand_control",
                executable="hand_control_node.py",
                name="hand_control_node",
                output="screen"
            ),
              
            # RELIABILITY TEST NODE
            Node(
                package="ingress",
                executable="reliability_node.py",
                name="reliability_node",
                output="log",
                parameters=[
                    {"motion_duration": 4.0},
                    {"recalibration_interval": 60.0},
                    {"flexion_scalar": 0.7},
                    {"retarget/hand_scheme": os.path.join(
                        get_package_share_directory("viz"),
                        "models",
                        "orca_v1",
                        "scheme_orca_v1.yaml",)
                    },
                ],
            ),
            
            # HAND DATA LOGGER NODE
            Node(
                package="ingress",
                executable="hand_data_logger_node.py",
                name="hand_data_logger",
                output="screen"
            ),
            
            # VISUALIZATION NODE
            
            Node(
                package="viz",
                executable="visualize_joints.py",
                name="visualize_joints",
                parameters=[
                    {
                        "scheme_path": os.path.join(
                            get_package_share_directory("viz"),
                            "models",
                            "orca_v1",
                            "scheme_orca_v1.yaml",
                        )
                    }
                ],
                output="screen",
            ),

            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                output='screen',
                parameters=[{'robot_description': robot_desc,}],
                arguments=[urdf]),
            
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen', 
                arguments=['-d', os.path.join(get_package_share_directory('viz'), 'rviz', 'retarget_config_orca_v1.rviz')],
                ),
                # Timer to shut down the launch after 1 hour
            TimerAction(
                period=10800.0,  # 3 hour in seconds
                actions=[Shutdown()]
                )
        ]
    )
