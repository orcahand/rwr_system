#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import yaml
import math
import time

class ReliabilityNode(Node):
    def __init__(self):
        super().__init__("joint_controller_node")
        self.publisher_ = self.create_publisher(Float32MultiArray, "/hand/policy_output", 10)
        
        self.motion_duration = self.declare_parameter("motion_duration", 10.0).value
        self.recalibration_interval = self.declare_parameter("recalibration_interval", 60.0).value
        self.flexion_scalar = self.declare_parameter("flexion_scalar", 1.0).value


        
        self.start_time = time.time()
        self.last_recalibration_time = self.start_time
        self.calibration_in_progress = False

        hand_scheme_path = self.declare_parameter("retarget/hand_scheme", "").value
        self.hand_scheme = self.load_hand_scheme(hand_scheme_path)
        
        self.gc_limits_lower, self.gc_limits_upper = self.get_joint_limits(self.hand_scheme)
        
        self.timer = self.create_timer(0.05, self.timer_callback)  # Update at 20 Hz

    def load_hand_scheme(self, path):
        if not path:
            raise ValueError("hand_scheme is required")
        with open(path, "r") as f:
            return yaml.safe_load(f)

     
    def get_joint_limits(self, hand_scheme):
        gc_limits_lower = np.deg2rad(np.array(hand_scheme["gc_limits_lower"]))
        gc_limits_upper = np.deg2rad(np.array(hand_scheme["gc_limits_upper"])) * self.flexion_scalar

        for index in [0, 1, 5, 8, 11, 14]:
            gc_limits_lower[index] = 0.0
            gc_limits_upper[index] = 0.0

        gc_limits_lower[2] = 0.35
        gc_limits_upper[2] = 0.35

        return gc_limits_lower, gc_limits_upper

    def timer_callback(self):
     
        current_time = time.time() - self.start_time
        elapsed_time_since_recalibration = time.time() - self.last_recalibration_time

        sine_wave = (-math.cos(current_time * (2 * math.pi / self.motion_duration)) + 1) / 2
        self.joint_values = (1 - sine_wave) * self.gc_limits_lower + sine_wave * self.gc_limits_upper

        if elapsed_time_since_recalibration >= self.recalibration_interval and np.allclose(self.joint_values, self.gc_limits_lower, atol=0.1):
            self.timer.cancel() # Cancel the timer

            # TODO ##############################################################
            self.get_logger().info("Stopping movement for initialization and auto-calibration")
            time.sleep(4)
            self.get_logger().info("Initialization and auto-calibration complete")
            # TODO ##############################################################

            self.last_recalibration_time = time.time()  # Reset the recalibration time
            self.start_time = time.time() - current_time # Reset the start time to continue smoothly
            self.timer = self.create_timer(0.05, self.timer_callback) # Restart the timer
            return

        msg = Float32MultiArray()
        msg.data = self.joint_values.tolist()
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ReliabilityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()