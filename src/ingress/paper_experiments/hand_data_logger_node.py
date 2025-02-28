#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger  # Import service
from std_msgs.msg import Float32MultiArray
import csv
import os
import datetime
import numpy as np

class HandDataLogger(Node):
    def __init__(self):
        super().__init__("hand_data_logger")

        # Generate timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        home_dir = os.path.expanduser("~")
        base_dir = os.path.join(home_dir, "ROS2_Logs", "data_logs")                       
                
        # Define subdirectories for different data types
        sub_dirs = {
            "current": os.path.join(base_dir, "current"),
            "temperature": os.path.join(base_dir, "temperature"),
            "pos_des": os.path.join(base_dir, "pos_des"),
            "pos_read": os.path.join(base_dir, "pos_read"),
            "policy_output": os.path.join(base_dir, "policy_output"),
            "sensing": os.path.join(base_dir, "sensing")
        }

        # Create directories if they don't exist
        for path in sub_dirs.values():
            os.makedirs(path, exist_ok=True)

        # Create filenames and open CSV files
        self.files = {}
        self.writers = {}
        for key, path in sub_dirs.items():
            filename = os.path.join(path, f"hand_{key}_{timestamp}.csv")
            self.files[key] = open(filename, "w", newline="")
            self.writers[key] = csv.writer(self.files[key])


        # Write headers with actual motor IDs
        header = ["timestamp"] + [f"motor_{i}" for i in range(17)]
        for writer in self.writers.values():
            writer.writerow(header)

        # Create subscribers
        self.current_sub = self.create_subscription(Float32MultiArray, "/hand/current", self.current_cb, 10)
        self.temperature_sub = self.create_subscription(Float32MultiArray, "/hand/temperature", self.temperature_cb, 10)
        self.pos_des_sub = self.create_subscription(Float32MultiArray, "/hand/pos_des", self.pos_des_cb, 10)
        self.pos_read_sub = self.create_subscription(Float32MultiArray, "/hand/pos_read", self.pos_read_cb, 10)
        self.policy_output_sub = self.create_subscription(Float32MultiArray, "/hand/policy_output", self.policy_output_cb, 10)
        self.sensing = self.create_subscription(Float32MultiArray, "/fsr_readings", self.sensing_cb, 10)


        self.get_logger().info(f"Data Logger node started. Saving logs in {base_dir}")

    def current_cb(self, msg: Float32MultiArray):
        rounded_data = [round(val, 2) for val in msg.data]
        self.write_csv_row(self.writers["current"], rounded_data)

    def temperature_cb(self, msg: Float32MultiArray):
        self.write_csv_row(self.writers["temperature"], msg.data)

    def pos_des_cb(self, msg: Float32MultiArray):
        rounded_data = [round(val, 4) for val in msg.data]
        self.write_csv_row(self.writers["pos_des"], rounded_data)

    def pos_read_cb(self, msg: Float32MultiArray):
        rounded_data = [round(val, 4) for val in msg.data]
        self.write_csv_row(self.writers["pos_read"], rounded_data)

    def policy_output_cb(self, msg: Float32MultiArray):
        self.write_csv_row(self.writers["policy_output"], msg.data)

    def sensing_cb(self, msg: Float32MultiArray):
        self.write_csv_row(self.writers["sensing"], msg.data)

    def write_csv_row(self, writer, data):
        timestamp = self.get_clock().now().nanoseconds / 1e9
        row = [timestamp] + list(data)
        writer.writerow(row)

    def destroy_node(self):
        """Close all open files before shutting down"""
        for file in self.files.values():
            file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
