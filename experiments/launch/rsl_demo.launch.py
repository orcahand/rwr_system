from launch import LaunchDescription
from launch_ros.actions import Node
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
            # ROKOKO INGRESS NODE
            Node(
                package="ingress",
                executable="rokoko_node.py",
                name="rokoko_node",
                output="log",
                parameters=[
                    {"rokoko_tracker/ip": "0.0.0.0"},
                    {"rokoko_tracker/port": 14043},
                    {"rokoko_tracker/use_coil": True}
                ],
            ),

            # HAND CONTROLLER NODE
            Node(
                package="hand_control",
                executable="hand_control_node.py",
                name="hand_control_node",
                output="screen"
            ),
            
            #RETARGET NODE
            Node(
                package="retargeter",
                executable="retargeter_node.py",
                name="retargeter_node",
                output="screen",
                # COMMENT OR UNCOMMENT THE FOLLOWING LINES TO SWITCH BETWEEN MJCF AND URDF, JUST ONE OF THEM SHOULD BE ACTIVE TODO: Make this a parameter
                parameters=[
                    {
                        "retarget/urdf_filepath": os.path.join(
                            get_package_share_directory("viz"),
                            "models",
                            "orca_v1",
                            "urdf",
                            "orca_v1_nowrist.urdf",
                        ),
                        "retarget/hand_scheme": os.path.join(
                            get_package_share_directory("viz"),
                            "models",
                            "orca_v1",
                            "scheme_orca_v1_nowrist.yaml",
                        ),
                        "retarget/mano_adjustments": os.path.join(
                            get_package_share_directory("experiments"),
                            "cfgs",
                            "retargeter_adjustment.yaml"
                        ),
                        "retarget/retargeter_cfg": os.path.join(
                            get_package_share_directory("experiments"),
                            "cfgs",
                            "retargeter_cfgs_orca_v1.yaml"
                        ),
                    },
                    {"debug": True},
                    {"include_wrist_and_tower": True},

                ],
            ),
            
        #     # VISUALIZATION NODE
        #     Node(
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

        ]
    )
