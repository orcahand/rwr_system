from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    urdf = os.path.join(
    get_package_share_directory('viz'),
    "models",
    "orca_v1",
    "urdf",
    "orca_v1.urdf")

    with open(urdf, 'r') as infp:
        robot_desc = infp.read()
     
    return LaunchDescription([
        
        # HAND CONTROLLER NODE
        # Node(
        #     package="hand_control",
        #     executable="hand_control_node.py",
        #     name="hand_control_node",
        #     output="screen"
        # ),

        Node(
            package='experiments',
            executable='run_rl_policy.py',
            name='angle_publisher',
            parameters=[
                    {
                         "policy_path": os.path.join(
                            get_package_share_directory('experiments'),
                            "cfgs",
                            "2025-02-27_04-00-51_dof_poses.npy",
                        )
                    },
                ],
            output='screen',
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

    ])