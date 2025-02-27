#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import os

def scale(values, lower_limits, upper_limits):
    """
    Scales the values to be within the specified lower and upper limits.
    """
    return lower_limits + (values + 1) * (upper_limits - lower_limits) / 2.0

class AnglePublisher(Node):
    def __init__(self):
        super().__init__('angle_publisher')

        self.declare_parameter("policy_path", rclpy.Parameter.Type.STRING)
        try:
            self.policy_path = self.get_parameter('policy_path').value
        except: 
            self.policy_path = None

            
        self.publisher_ = self.create_publisher(Float32MultiArray, 'hand/policy_output', 10)
        timer_period = 1.0 / 60.0  # 60 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.obs = False
        # Load the npy file

        

        if self.obs:
            file_path = os.path.expanduser("~/Downloads/recs/2025-02-27_02-52-43_observation.npy")
        else:
            file_path = os.path.expanduser("~/Downloads/rec2/2025-02-27_04-00-51_dof_poses.npy")
        
        print(f"Loading file from: {file_path}")
        self.data = np.load(file_path)
        print(f"Loaded data shape: {self.data.shape}")
        
        self.index = 0
        self.angles = np.zeros(17)
        self.smoothed_data = self.smooth_data(self.data)

        # Define joint range limits
        self.joint_range_of_policy = np.array([
            [-0.79, -0.3], 
            [-0.5, 0.5], 
            [0.3, 1.5], 
            [-0, 1.5],
            [-0.5, 0.5], 
            [0.3, 1.5], 
            [-0, 1.5],
            [-0.5, 0.5], 
            [0.3, 1.5], 
            [-0, 1.5],
            [-0.5, 0.5], 
            [0.3, 1.5], 
            [-0, 1.5], 
            [-0.1, 0.1],  
            [-0.7853981633974483, 0.7853981633974483], 
            [-0.0, 1.4], 
            [-0.0, 0.79],   
        ])

        self.lower_limit, self.upper_limit = self.joint_range_of_policy[:,0], self.joint_range_of_policy[:,1]

    def smooth_data(self, data):
        """
        Inserts averaged angle values between two elements for all 1200 elements.
        """
        smoothed_data = []
        for j in range(data.shape[0]):
            for i in range(data.shape[1] - 1):
                smoothed_data.append(data[j, i, :])
                averaged_angles_mid = (data[j, i, :] + data[j, i + 1, :]) / 2.0
                averaged_angles_first = (data[j, i, :] + averaged_angles_mid) / 2.0
                averaged_angles_last = (averaged_angles_mid + data[j, i + 1, :]) / 2.0
                smoothed_data.append(averaged_angles_first)
                smoothed_data.append(averaged_angles_mid)
                smoothed_data.append(averaged_angles_last)
            smoothed_data.append(data[j, -1, :])  # Append the last element
        return np.array(smoothed_data)

    def sim_to_real(self, input_array):
        """
        Reorders the elements of the input array according to the inverse of the specified mapping.
        """
        inverse_mapping = [0, 13, 14, 15, 16, 10, 11, 12, 4, 5, 6, 1, 2, 3, 7, 8, 9]
        reordered_array = [input_array[i] for i in inverse_mapping]
        return np.array(reordered_array)

    def timer_callback(self):
        print(self.smoothed_data.shape)
        if self.index >= self.smoothed_data.shape[0]:
            self.index = 0  # Loop back to the start if we reach the end

        if self.obs:
            angles = self.smoothed_data[self.index, -17:]
            self.angles += angles * 0.167
        else:
            angles = self.smoothed_data[self.index, :]
            self.angles = angles.flatten()

        # Ensure angles are within the specified limits
        self.angles = scale(self.angles, self.lower_limit, self.upper_limit)
        self.angles = np.clip(self.angles, self.lower_limit, self.upper_limit)

        self.angles = self.sim_to_real(self.angles)

        msg = Float32MultiArray()
        msg.data = self.angles.tolist()

        print(f"Publishing angles: {self.angles}")
        self.publisher_.publish(msg)

        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    angle_publisher = AnglePublisher()
    rclpy.spin(angle_publisher)
    angle_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()