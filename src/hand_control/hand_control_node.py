#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import Trigger  # Import Trigger service type
import os
from faive_system.src.hand_control.hand_controller import OrcaHand

class HandControllerNode(Node):
    def __init__(self, debug=False):
        super().__init__("hand_controller_node")
        self.get_logger().info("Hand Controller Node Started")

        # start tracker
        self.declare_parameter("hand_controller/port", "/dev/ttyUSB0")
        self.declare_parameter("hand_controller/baudrate", 3000000)

        port = self.get_parameter("hand_controller/port").value
        baudrate = self.get_parameter("hand_controller/baudrate").value

        self._hc = OrcaHand(model_path=None)
        self._hc.connect()
        self._hc.init_joints()

        # Subscribers
        self.joint_angle_sub = self.create_subscription(Float32MultiArray, "/hand/policy_output", self.joint_angle_cb, 10)

        # Publishers
        self.curr_pub = self.create_publisher(Float32MultiArray, "/hand/current", 10)
        self.temp_pub = self.create_publisher(Float32MultiArray, "/hand/temperature", 10)
        # self.pos_desired_pub = self.create_publisher(Float32MultiArray, "/hand/pos_des", 10)
        self.pos_read_pub = self.create_publisher(Float32MultiArray, "/hand/pos_read", 10)

        # Timer: run at 20 Hz => every 0.05 s
        self.monitor_timer = self.create_timer(0.05, self.monitor_callback)

    def joint_angle_cb(self, msg):
        assert len(msg.data) == 17, "Expected 17 joint angles, got {}".format(
            len(msg.data)
        )
        joint_angles = np.array(msg.data)
        joint_angles_deg = joint_angles * 180 / np.pi
        self._hc.set_mano_points(joint_angles_deg)


    def monitor_callback(self):
        """Runs at 20Hz to publish each motor’s current & temperature."""
        # Get current in mA
        motor_currents = self._hc.get_motor_current()  # shape: (num_motors,)

        # Get temperature in °C as uint8
        motor_temps = self._hc.get_motor_temp()  # shape: (num_motors,)

        motor_pos = self._hc.get_motor_pos()  # shape: (num_motors,)


        # Convert data for publishing
        curr_msg = Float32MultiArray(data=motor_currents.tolist())
        temp_msg = Float32MultiArray(data=motor_temps.tolist())
        pos_msg = Float32MultiArray(data=motor_pos.tolist())

        self.curr_pub.publish(curr_msg)
        self.temp_pub.publish(temp_msg)
        self.pos_read_pub.publish(pos_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HandControllerNode()
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
