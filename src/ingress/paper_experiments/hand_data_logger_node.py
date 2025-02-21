#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import csv
import os
import datetime

class HandDataLogger(Node):
    def __init__(self):
        super().__init__("hand_data_logger")

        # Generate timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H")

        # Define base directory where logs will be saved
        base_dir = os.path.expanduser("~/data_logs")  

        # Define subdirectories for different data types
        sub_dirs = {
            "current": os.path.join(base_dir, "current"),
            "temperature": os.path.join(base_dir, "temperature"),
            "pos_des": os.path.join(base_dir, "pos_des"),
            "pos_read": os.path.join(base_dir, "pos_read"),
            "policy_output": os.path.join(base_dir, "policy_output")
        }

        # Create directories if they don't exist
        for path in sub_dirs.values():
            os.makedirs(path, exist_ok=True)

        # Open CSV files in their respective subdirectories
        self.current_file = open(os.path.join(sub_dirs["current"], f"hand_current_{timestamp}.csv"), "w", newline="")
        self.temperature_file = open(os.path.join(sub_dirs["temperature"], f"hand_temperature_{timestamp}.csv"), "w", newline="")
        self.pos_des_file = open(os.path.join(sub_dirs["pos_des"], f"hand_pos_des_{timestamp}.csv"), "w", newline="")
        self.pos_read_file = open(os.path.join(sub_dirs["pos_read"], f"hand_pos_read_{timestamp}.csv"), "w", newline="")
        self.policy_file = open(os.path.join(sub_dirs["policy_output"], f"hand_policy_output_{timestamp}.csv"), "w", newline="")

        # Create CSV writers
        self.current_writer = csv.writer(self.current_file)
        self.temperature_writer = csv.writer(self.temperature_file)
        self.pos_des_writer = csv.writer(self.pos_des_file)
        self.pos_read_writer = csv.writer(self.pos_read_file)
        self.policy_writer = csv.writer(self.policy_file)

        # Write headers
        header = ["timestamp"] + [f"motor_{i}" for i in range(17)]
        self.current_writer.writerow(header)
        self.temperature_writer.writerow(header)
        self.pos_des_writer.writerow(header)
        self.pos_read_writer.writerow(header)
        self.policy_writer.writerow(header)

        # Create subscribers
        self.current_sub = self.create_subscription(Float32MultiArray, "/hand/current", self.current_cb, 10)
        self.temperature_sub = self.create_subscription(Float32MultiArray, "/hand/temperature", self.temperature_cb, 10)
        self.pos_des_sub = self.create_subscription(Float32MultiArray, "/hand/pos_des", self.pos_des_cb, 10)
        self.pos_read_sub = self.create_subscription(Float32MultiArray, "/hand/pos_read", self.pos_read_cb, 10)
        self.policy_output_sub = self.create_subscription(Float32MultiArray, "/hand/policy_output", self.policy_output_cb, 10)

        self.get_logger().info(f"Data Logger node started. Saving logs in {base_dir}")

    def current_cb(self, msg: Float32MultiArray):
        self.write_csv_row(self.current_writer, msg)

    def temperature_cb(self, msg: Float32MultiArray):
        self.write_csv_row(self.temperature_writer, msg)

    def pos_des_cb(self, msg: Float32MultiArray):
        self.write_csv_row(self.pos_des_writer, msg)

    def pos_read_cb(self, msg: Float32MultiArray):
        self.write_csv_row(self.pos_read_writer, msg)

    def policy_output_cb(self, msg: Float32MultiArray):
        self.write_csv_row(self.policy_writer, msg)

    def write_csv_row(self, writer, msg: Float32MultiArray):
        timestamp = self.get_clock().now().nanoseconds / 1e9
        row = [timestamp] + list(msg.data)
        writer.writerow(row)

    def destroy_node(self):
        self.current_file.close()
        self.temperature_file.close()
        self.pos_des_file.close()
        self.pos_read_file.close()
        self.policy_file.close()
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
