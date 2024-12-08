#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
import yaml
import os
import time
import math
import tf2_ros
import tf2_geometry_msgs


class VisualizeJointsNode(Node):
    def __init__(self):
        super().__init__("visualize_joints_node")
        self.subscription = self.create_subscription(
            Float32MultiArray, "/hand/policy_output", self.policy_output_callback, 10
        )
        self.publisher_ = self.create_publisher(JointState, "/joint_states", 10)
        self.marker_publisher = self.create_publisher(MarkerArray, "/sensor_visualization_marker", 10)
        self.get_logger().info('Subscribing to "/hand/policy_output"')
        self.get_logger().info('Publishing to "/joint_states"')
        self.declare_parameter('scheme_path', "")
        scheme_path = self.get_parameter("scheme_path").value
        print(f"Reading hand scheme from {scheme_path}")
        # Read the YAML file directly
        with open(scheme_path, 'r') as f:
            self.hand_scheme = yaml.safe_load(f)

        self.tendons = self.hand_scheme["gc_tendons"]
        # a list representation of the jacobian matrix. Each element is a tuple (tendon_name, factor)
        self.jacobian_list = []
        self.joint_names = []
        for i, (tendon_name, joints) in enumerate(self.tendons.items()):
            self.joint_names.append(tendon_name)
            if "orca" in os.path.basename(scheme_path):
                pin_joint_contact_factor = 1.0  # No scaling, ideal pin joint behavior
                self.jacobian_list.append((i, pin_joint_contact_factor))  # No change in factor
                for joint_name, factor in joints.items():
                    self.joint_names.append(joint_name)
                    self.jacobian_list.append((i, factor * pin_joint_contact_factor))  # The factor remains the same
            else:
                rolling_contact_factor = 0.5 if tendon_name.endswith("_virt") else 1.0
                self.jacobian_list.append((i, rolling_contact_factor))
                for joint_name, factor in joints.items():
                    self.joint_names.append(joint_name)
                    self.jacobian_list.append((i, factor * rolling_contact_factor))
                

        self.js_msg = JointState()
        self.js_msg.name = self.joint_names

        self.marker_array = MarkerArray()
        fingertip_names = ["thumb_fingertip", "index_fingertip", "middle_fingertip", "ring_fingertip", "pinky_fingertip"]
        for i, fingertip_name in enumerate(fingertip_names):
            marker = Marker()
            marker.header.frame_id = fingertip_name
            marker.ns = "fingertips"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.pose.position = Point(x=0.0, y=0.005, z=-0.01)
            marker.pose.orientation.w = 1.0
            self.marker_array.markers.append(marker)

        # Timer to update the marker colors based on the sine wave
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_time = time.time()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


    def timer_callback(self):
        current_time = time.time() - self.start_time
        intensity = (math.sin(current_time) + 1) / 2  # Normalize sine wave to [0, 1]
        # color = self.calculate_color(intensity)
        # opacity = intensity
        # for marker in self.marker_array.markers:
        #     # marker.color.r = color[0]
        #     # marker.color.g = color[1]
        #     # marker.color.b = color[2]
        #     # marker.color.a = color[3]
        #     marker.color.a = opacity
        #     marker.color.r = 1.0
        #     marker.color.g = 0.0
        #     marker.color.b = 0.0# Get the position of the thumb_fingertip
        try:
            # Get the transform from each fingertip frame to the hand_root frame
            thumb_transform = self.tf_buffer.lookup_transform('hand_root', 'thumb_fingertip', rclpy.time.Time())
            thumb_point = PointStamped()
            thumb_point.header.frame_id = 'thumb_fingertip'
            thumb_point.point = Point(x=0.0, y=0.005, z=-0.01)
            thumb_position = tf2_geometry_msgs.do_transform_point(thumb_point, thumb_transform).point

            max_opacity = 0.0

            for marker in self.marker_array.markers:
                if marker.header.frame_id != "thumb_fingertip":
                    fingertip_transform = self.tf_buffer.lookup_transform('hand_root', marker.header.frame_id, rclpy.time.Time())
                    fingertip_point = PointStamped()
                    fingertip_point.header.frame_id = marker.header.frame_id
                    fingertip_point.point = marker.pose.position
                    fingertip_position = tf2_geometry_msgs.do_transform_point(fingertip_point, fingertip_transform).point

                    # Calculate the distance between the thumb_fingertip and the current fingertip
                    distance = math.sqrt(
                        (thumb_position.x - fingertip_position.x) ** 2 +
                        (thumb_position.y - fingertip_position.y) ** 2 +
                        (thumb_position.z - fingertip_position.z) ** 2
                    )

                    threshold = 0.03
                    if distance < threshold:
                        opacity = 1.0 - (distance) ** 2
                    else:
                        opacity = 0.0

                    marker.color.a = opacity
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0

                    # Update the maximum opacity for the thumb marker
                    max_opacity = max(max_opacity, opacity)


            thumb_fingertip_marker = next(marker for marker in self.marker_array.markers if marker.header.frame_id == "thumb_fingertip")
            thumb_fingertip_marker.color.a = max_opacity
            thumb_fingertip_marker.color.r = 1.0
            thumb_fingertip_marker.color.g = 0.0
            thumb_fingertip_marker.color.b = 0.0

            self.marker_publisher.publish(self.marker_array)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Transform error: {e}")


    def calculate_color(self, intensity):
        # Map intensity to color (e.g., red to green)
        r = max(0.0, min(1.0, 1.0 - intensity))
        g = max(0.0, min(1.0, intensity))
        b = 0.0
        return [r, g, b, 1.0]

    def policy_output_callback(self, msg):
        
        self.js_msg.header.stamp = self.get_clock().now().to_msg()
        joint_states = self.policy_output2urdf_joint_states(msg.data)
        self.js_msg.position = joint_states
        self.publisher_.publish(self.js_msg)
        # self.get_logger().info('Publishing joint states: "%s"' % self.js_msg)

    def policy_output2urdf_joint_states(self, joint_values):
        """
        Process joint values to create a vector where each joint's value is halved.
        If the joint has a corresponding virtual joint (virt), duplicate the value.

        :param joint_values: List of joint values with length N (in this case, 16).
        :param has_virt_joint: List indicating if each joint has a virtual counterpart.
        :return: List with processed joint values, doubled for those with virtual joints.
        """
        # Initialize the output list for processed joint values
        assert (len(joint_values) == len(self.tendons)), f"The length of joint values {len(joint_values)} should match the number of tendons {len(self.tendons)}"
        # Iterate over each joint value and corresponding virtual status
        processed_values = [joint_values[ind] * factor for ind, factor in self.jacobian_list]
        return processed_values


def main(args=None):
    rclpy.init(args=args)
    node = VisualizeJointsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
