#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import Trigger
import numpy as np
import yaml
import math
import time

class AccuracyNode(Node):
    def __init__(self):
        super().__init__("accuracy_node")
        self.publisher_ = self.create_publisher(Float32MultiArray, "/hand/policy_output", 10)
        
        self.motion_duration = self.declare_parameter("motion_duration", 10.0).value
        self.recalibration_interval = self.declare_parameter("recalibration_interval", 60.0).value
        self.flexion_scalar = self.declare_parameter("flexion_scalar", 1.0).value
        self.signal_type = self.declare_parameter("signal_type", "sine").value  # New parameter for signal type
        self.calibration = self.declare_parameter("calibration", False).value

        self.start_time = time.time()
        self.last_recalibration_time = self.start_time

        hand_scheme_path = self.declare_parameter("retarget/hand_scheme", "").value
        self.hand_scheme = self.load_hand_scheme(hand_scheme_path)
        
        self.gc_limits_lower, self.gc_limits_upper = self.get_joint_limits(self.hand_scheme)
        
        
        self.initial_Calibration = False
        self.awaiting_response = False
        self.calib_client = self.create_client(Trigger, "/hand/start_auto_calib")
        
        self.timer = self.create_timer(0.01, self.timer_callback)  # Update at 100 Hz

    def load_hand_scheme(self, path):
        if not path:
            raise ValueError("hand_scheme is required")
        with open(path, "r") as f:
            return yaml.safe_load(f)

     
    def get_joint_limits(self, hand_scheme):
        gc_limits_lower = np.deg2rad(np.array(hand_scheme["gc_limits_lower"]))
        gc_limits_upper = np.deg2rad(np.array(hand_scheme["gc_limits_upper"])) * self.flexion_scalar

        for index in range(len(gc_limits_lower)):
            if index not in [6, 7]:
                gc_limits_lower[index] = 0.0
                gc_limits_upper[index] = 0.0

        # move the thumb away, such that markers on index can be seen better (watch out - hard coded for old model)
        # gc_limits_lower[1] = np.deg2rad(45) 
        # gc_limits_upper[1] = np.deg2rad(45)
        # gc_limits_lower[2] = np.deg2rad(40)
        # gc_limits_upper[2] = np.deg2rad(40)

        return gc_limits_lower, gc_limits_upper

    def timer_callback(self):
     
        current_time = time.time() - self.start_time
        elapsed_time_since_recalibration = time.time() - self.last_recalibration_time

        if self.signal_type == "sine":
            sine_wave = (-math.cos(current_time * (2 * math.pi / self.motion_duration)) + 1) / 2
            self.joint_values = (1 - sine_wave) * self.gc_limits_lower + sine_wave * self.gc_limits_upper
        elif self.signal_type == "step":
            step_wave = 1 if (current_time % self.motion_duration) < (self.motion_duration / 2) else 0
            self.joint_values = (1 - step_wave) * self.gc_limits_lower + step_wave * self.gc_limits_upper

        if self.calibration and (self.initial_Calibration or (elapsed_time_since_recalibration >= self.recalibration_interval) and np.allclose(self.joint_values, self.gc_limits_lower, atol=0.1)):
            self.timer.cancel() # Cancel the timer

            if not self.awaiting_response:  # Only request if no response is pending
                self.initial_Calibration = False
                self.request_calibration()

            self.last_recalibration_time = time.time()  # Reset the recalibration time
            self.start_time = time.time() - current_time # Reset the start time to continue smoothly
            self.timer = self.create_timer(0.05, self.timer_callback) # Restart the timer

        if not self.awaiting_response:
            msg = Float32MultiArray()
            msg.data = self.joint_values.tolist()
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
    node = AccuracyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()