#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import Trigger
import numpy as np
import yaml
import math
import time

class ReliabilityNode(Node):
    def __init__(self):
        super().__init__("reliability_node")
        self.publisher_ = self.create_publisher(Float32MultiArray, "/hand/policy_output", 10)
    
        self.motion_duration = self.declare_parameter("motion_duration", 10.0).value
        self.recalibration_interval = self.declare_parameter("recalibration_interval", 60.0).value
        self.flexion_scalar = self.declare_parameter("flexion_scalar", 1.0).value
        
        self.start_time = time.time()
        self.last_recalibration_time = self.start_time

        hand_scheme_path = self.declare_parameter("retarget/hand_scheme", "").value
        self.hand_scheme = self.load_hand_scheme(hand_scheme_path)
        
        self.gc_limits_lower, self.gc_limits_upper = self.get_joint_limits(self.hand_scheme)
    
        self.initial_Calibration = True
        self.awaiting_response = False
        self.calib_client = self.create_client(Trigger, "/hand/start_auto_calib")
        
        # self.timer = self.create_timer(0.05, self.timer_callback)  # Update at 20 Hz
        self.timer = self.create_timer(0.02, self.timer_callback)  # Update at 100 Hz

    def load_hand_scheme(self, path):
        if not path:
            raise ValueError("hand_scheme is required")
        with open(path, "r") as f:
            return yaml.safe_load(f)

     
    def get_joint_limits(self, hand_scheme):
        gc_limits_lower = np.deg2rad(np.array(hand_scheme["gc_limits_lower"]))
        gc_limits_upper = np.deg2rad(np.array(hand_scheme["gc_limits_upper"])) * self.flexion_scalar

        for index in [1, 2, 3, 4,  5, 8, 11, 14]: # Wrist 0
            gc_limits_lower[index] = 0.0
            gc_limits_upper[index] = 0.0

        gc_limits_lower[0] *= self.flexion_scalar
        # gc_limits_upper[0] = 40.0

        return gc_limits_lower, gc_limits_upper

    def timer_callback(self):
        current_time = time.time() - self.start_time

        # Normal sine wave for non-wrist joints
        sine_wave = (-math.cos(current_time * (2 * math.pi / self.motion_duration)) + 1) / 2
        joint_values = (1 - sine_wave) * self.gc_limits_lower + sine_wave * self.gc_limits_upper

        # Calculate wrist sine wave with half frequency (period doubled)
        wrist_sine_wave = (-math.cos(current_time * (2 * math.pi / (4 * self.motion_duration))) + 1) / 2
        # Override joint 0 (wrist) value using its own lower and upper limits
        joint_values[0] = (1 - wrist_sine_wave) * self.gc_limits_lower[0] + wrist_sine_wave * self.gc_limits_upper[0]

        msg = Float32MultiArray()
        msg.data = joint_values.tolist()
        self.publisher_.publish(msg)

    def request_calibration(self):
        """Sends a request to the hand_control_node to start auto-calibration."""
        if not self.calib_client.wait_for_service(timeout_sec=6.0):
            self.get_logger().error("Calibration service is unavailable.")
            return
        
        self.get_logger().info("Requesting auto-calibration...")
        request = Trigger.Request()
        future = self.calib_client.call_async(request)
        future.add_done_callback(self.calibration_response)
        self.awaiting_response = True  # Prevent duplicate requests

    def calibration_response(self, future):
        """Handles the response from the calibration service."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Auto-calibration completed successfully.")
                self.last_recalibration_time = time.time()  # Update the last recalibration time
            else:
                self.get_logger().error(f"Auto-calibration failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        
        self.awaiting_response = False  # Allow new requests

def main(args=None):
    rclpy.init(args=args)
    node = ReliabilityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()