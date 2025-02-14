#!/bin/env python3

import os
import glob
import argparse
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import numpy as np
import h5py
#from logger_node import TOPICS_TYPES  # Import the predefined topic types
import cv2
from logger_node import TOPICS_TYPES  # Import the predefined topic types
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

TOPICS_TYPES = {
    # FRANKA ROBOT
    "/franka/end_effector_pose": PoseStamped,
    "/franka/end_effector_pose_cmd": PoseStamped,
    
    # HAND POLICY OUTPUT
    "/hand/policy_output": Float32MultiArray,
    
    # CAMERA IMAGES
    "/oakd_front_view/color": Image,
    "/oakd_side_view/color": Image,
    "/oakd_wrist_view/color": Image,
    
    "/task_description": String,  # New topic for task description
    
    # CAMERA PARAMETERS
    "/oakd_front_view/intrinsics": Float32MultiArray,
    "/oakd_side_view/intrinsics": Float32MultiArray,
    "/oakd_wrist_view/intrinsics": Float32MultiArray,
    "/oakd_front_view/extrinsics": Float32MultiArray,
    "/oakd_side_view/extrinsics": Float32MultiArray,
    "/oakd_wrist_view/extrinsics": Float32MultiArray,
    "/oakd_front_view/projection": Float32MultiArray,
    "/oakd_side_view/projection": Float32MultiArray,
    "/oakd_wrist_view/projection": Float32MultiArray,
}


TOPIC_TO_STRING = {
    Float32MultiArray: "Float32MultiArray",
    PoseStamped: "PoseStamped",
    Image: "Image",
    String: "String",
}

def get_topic_names(h5_path):
    with h5py.File(h5_path, 'r') as h5_file:
        topic_names = list(h5_file.keys())
        print(f"Topics in the HDF5 file: {topic_names}")
    return topic_names

def sample_and_sync_h5(input_h5_path, output_h5_path, sampling_frequency, compress, resize_to, topic_types):
    qpos_franka = None
    qpos_hand = None
    actions_franka = None
    actions_hand = None
    """
    Sample images and interpolate data for synchronization.
    
    Parameters:
        input_h5_path (str): Path to the input HDF5 file.
        output_h5_path (str): Path to the output HDF5 file.
        sampling_frequency (float): Sampling frequency in Hz.
        topic_types (dict): Dictionary mapping topics to their types.
    """
    with h5py.File(input_h5_path, 'r') as input_h5, h5py.File(output_h5_path, 'w') as output_h5:
        # Determine sampling timestamps
        start_time = None
        end_time = None
        for topic in topic_types:
            if topic in input_h5:
                if topic == "/task_description":
                    continue
                timestamps = np.array(list(map(int, input_h5[topic].keys())))
                if start_time is None or timestamps[0] < start_time:
                    start_time = timestamps[0]
                if end_time is None or timestamps[-1] > end_time:
                    end_time = timestamps[-1]

        desired_timestamps = np.arange(
            start_time, end_time, 1e9 / sampling_frequency
        ).astype(int)

        # Process each topic
        for topic, topic_type in topic_types.items():
            if topic not in input_h5:
                print(f"Topic {topic} not found in the HDF5 file. Skipping...")
                continue
            
            
            print(f"Processing topic: {topic}")
            topic_group = input_h5[topic]

            if topic == "/task_description":
                if TOPIC_TO_STRING[topic_type] == "String":
                    string_data = topic_group["description"]
                    output_h5.create_dataset("task_description", data=string_data)
                continue


            topic_timestamps = np.array(list(map(int, topic_group.keys())))
            topic_timestamps.sort()

            if "intrinsics" in topic or "extrinsics" in topic or "projection" in topic:
                data = np.array(topic_group[str(topic_timestamps[0])][:])
                output_h5.create_dataset(f"observations/images/{topic}", data=data)
                continue

            if TOPIC_TO_STRING[topic_type] == "Image":
                # Sample images
                sampled_images = []
                for t in desired_timestamps:
                    closest_idx = np.abs(topic_timestamps - t).argmin()
                    closest_timestamp = topic_timestamps[closest_idx]
                    sampled_images.append(topic_group[str(closest_timestamp)][:])

                if resize_to is not None:
                    sampled_images = [cv2.resize(img, resize_to, interpolation=cv2.INTER_LINEAR) for img in sampled_images]

                sampled_images = np.array(sampled_images)  # TxHxWxC
                chunk_size = (1,) + tuple(sampled_images.shape[1:])
                if compress:
                    output_h5.create_dataset(f"observations/images/{topic}", data=sampled_images, chunks = chunk_size, compression="lzf")
                else:
                    output_h5.create_dataset(f"observations/images/{topic}", data=sampled_images, chunks = chunk_size)

            elif TOPIC_TO_STRING[topic_type] == "PoseStamped":
                # Interpolate PoseStamped data
                pose_data = np.array([topic_group[str(ts)][:] for ts in topic_timestamps])
                positions = pose_data[:, :3]
                quaternions = pose_data[:, 3:]
                
                interp_position = interp1d(
                    topic_timestamps, positions, axis=0, kind="linear", fill_value="extrapolate"
                )
                interp_quaternions = interp1d(
                    topic_timestamps, quaternions, axis=0, kind="linear", fill_value="extrapolate"
                )

                sampled_positions = interp_position(desired_timestamps)
                sampled_quaternions = interp_quaternions(desired_timestamps)
                sampled_quaternions /= np.linalg.norm(
                    sampled_quaternions, axis=1, keepdims=True
                )  # Normalize quaternions
                
                if topic == "/franka/end_effector_pose":
                    qpos_franka = np.concatenate((sampled_positions, sampled_quaternions), axis=1)
                elif topic == "/franka/end_effector_pose_cmd":
                    actions_franka = np.concatenate((sampled_positions, sampled_quaternions), axis=1)


            elif TOPIC_TO_STRING[topic_type] == "Float32MultiArray":
                # Interpolate Float32MultiArray data
                array_data = np.array([topic_group[str(ts)][:] for ts in topic_timestamps])
                interp_array = interp1d(
                    topic_timestamps, array_data, axis=0, kind="linear", fill_value="extrapolate"
                )
                sampled_array = interp_array(desired_timestamps)
                
                qpos_hand = sampled_array
                actions_hand = sampled_array
            
        
            # create observations group
        if qpos_franka is not None:
            output_h5.create_dataset("observations/qpos_franka", data=qpos_franka)
        if qpos_hand is not None:
            output_h5.create_dataset("observations/qpos_hand", data=qpos_hand)
        if actions_franka is not None:
            output_h5.create_dataset("actions_franka", data=actions_franka)
        if actions_hand is not None:
            output_h5.create_dataset("actions_hand", data=actions_hand)



    print(f"Processed data saved to: {output_h5_path}")

def process_folder(input_folder, sampling_frequency, compress, resize_to, topic_types):
    """
    Process all HDF5 files in the given folder and save the processed files
    with a running index in a new folder named <input_folder>_processed.
    
    Parameters:
        input_folder (str): Path to the folder containing input HDF5 files.
        sampling_frequency (float): Sampling frequency in Hz.
        topic_types (dict): Dictionary mapping topics to their types.
    """
    # Get all HDF5 files in the folder
    h5_files = sorted(glob.glob(os.path.join(input_folder, "*.h5")))
    if not h5_files:
        print(f"No HDF5 files found in {input_folder}.")
        return

    # Create the output folder
    output_folder = os.path.join(os.path.dirname(input_folder), 
                                 os.path.basename(input_folder) + "_processed" + f"_{int(sampling_frequency)}hz")
    if compress:
        output_folder += "_lzf"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created: {output_folder}")

    # Process each file
    for idx, input_file in enumerate(h5_files):
        try:
            output_file = os.path.join(output_folder, f"{idx:04d}.h5")
            print(f"Processing file: {input_file}")
            sample_and_sync_h5(input_file, output_file, sampling_frequency, compress, resize_to, topic_types)
            print(f"Processed file saved as: {output_file}")
        except Exception as e:
            print(e)

    print(f"All files processed. Processed files are saved in {output_folder}.")

def main():
    parser = argparse.ArgumentParser(description="Process and synchronize HDF5 files.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing input HDF5 files.")
    # parser.add_argument("--sampling_freq", type=float, default=100, help="Sampling frequency in Hz.")
    parser.add_argument("--sampling_freq", type=float, default=20, help="Sampling frequency in Hz.")
    parser.add_argument("--sampling_freq", type=float, default=100, help="Sampling frequency in Hz.")
    parser.add_argument("--compress",  action="store_true", help="Compress the output HDF5 files. [it might boost the performance on aws but might decrease the performance on local machine]")
    parser.add_argument(
        '--resize_to',
        type=lambda s: tuple(map(int, s.strip("()").split(","))),
        help="Target size of the image as a tuple of integers, e.g., '(width, height)'.",
        default=None
    )
    args = parser.parse_args()

    # Process all files in the folder
    process_folder(args.input_folder, args.sampling_freq, args.compress, args.resize_to, TOPICS_TYPES)

if __name__ == "__main__":
    main()



#!/usr/bin/env python3

import rospy
import yaml
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from apriltag_ros.msg import AprilTagDetectionArray
from pyquaternion import Quaternion as PyQuaternion
import numpy as np
from nav_msgs.msg import Odometry

class VizNode:
    def __init__(self):
        rospy.init_node('viz_node', anonymous=True)
        
        self.marker_pub = rospy.Publisher('global_map_viz', MarkerArray, queue_size=10)
        self.duckiebot_pub = rospy.Publisher('duckiebot_viz', Marker, queue_size=10)

        self.tag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.self_localization_callback)    
        self.odom_sub = rospy.Subscriber(f'/wheel_encoder/odom', Odometry, self.odom_callback)


        self.yaml_file = rospy.get_param('~yaml_file', '/code/catkin_ws/src/user_code/quack-norris/params/apriltags.yaml')
        self.maps_yaml_file = rospy.get_param('~maps_yaml_file', '/code/catkin_ws/src/user_code/quack-norris/params/maps.yaml')
        self.map_name = rospy.get_param('~map_name', 'lap_big')
        
        rospy.loginfo(f"Selected map: {self.map_name}")
        
        self.maps = self.load_maps(self.maps_yaml_file)
        self.map_params = self.get_map_params(self.maps, self.map_name)
        self.qx = PyQuaternion(np.sqrt(0.5), np.sqrt(0.5), 0, 0) # 90 degree rot around x-axis
        self.camera_angle = PyQuaternion(0.5, 0.8660254, 0.0, 0.0) # rot around x-axis corresponding to camera angle on duckiebot
        self.qz = PyQuaternion(np.sqrt(0.5), 0, 0, np.sqrt(0.5)) # 90 degree rot around z-axis
        
        if not self.map_params:
            rospy.logerr(f"Map '{self.map_name}' not found in {self.maps_yaml_file}")
            return
                
        self.waypoints = self.load_waypoints(self.yaml_file)

        self.marker_array = MarkerArray()
        self.create_markers()
        
        self.duckiebot_marker = Marker()
        self.duckiebot_marker.header.frame_id = "map"
        self.duckiebot_marker.header.stamp = rospy.Time.now()
        self.duckiebot_marker.ns = "duckiebot"
        self.duckiebot_marker.id = 0
        self.duckiebot_marker.type = Marker.MESH_RESOURCE
        self.duckiebot_marker.action = Marker.ADD
        self.duckiebot_marker.mesh_resource = "file:///code/catkin_ws/src/user_code/quack-norris/map_files/duckiebot-blue.dae"
        self.duckiebot_marker.mesh_use_embedded_materials = True
        z_offset_mesh = 0.0335  # offset such that the wheels are on the ground
        self.duckiebot_marker.pose.position = Point(0, 0, z_offset_mesh)
        self.duckiebot_marker.pose.orientation = Quaternion(0, 0, 0, 1)
        self.duckiebot_marker.scale = Vector3(1, 1, 1)
        
        self.use_apriltag = False
        self.initial_pose_set = False
        self.initial_duckie_marker_pose = None
        self.initial_duckie_marker_orientation = None
        self.initial_odom_pose = None
        self.initial_odom_orientation = None



    def load_maps(self, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        return data['maps']

    def get_map_params(self, maps, map_name):
        for map_entry in maps:
            if map_entry['name'] == map_name:
                return map_entry
        return None

    def load_waypoints(self, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        return data['standalone_tags']

    def create_markers(self):
        plane_width = self.map_params['plane_width']
        mesh_path = self.map_params['mesh_path']
        
        # Create the marker for the map
        map_marker = Marker()
        map_marker.header.frame_id = "map"
        map_marker.header.stamp = rospy.Time.now()
        map_marker.ns = "map"
        map_marker.id = 0
        map_marker.type = Marker.MESH_RESOURCE
        map_marker.action = Marker.ADD
        map_marker.pose.position.x = -plane_width / 2.0
        map_marker.pose.position.y = -plane_width / 2.0
        map_marker.pose.position.z = 0.0
        map_marker.pose.orientation.x = 0.0
        map_marker.pose.orientation.y = 0.0
        map_marker.pose.orientation.z = 0.0
        map_marker.pose.orientation.w = 1.0
        map_marker.scale.x = 1.0
        map_marker.scale.y = 1.0
        map_marker.scale.z = 1.0
        map_marker.color.a = 1.0
        map_marker.color.r = 1.0
        map_marker.color.g = 1.0
        map_marker.color.b = 1.0
        map_marker.mesh_resource = mesh_path
        map_marker.mesh_use_embedded_materials = True

        self.marker_array.markers.append(map_marker)
        
        for waypoint in self.waypoints:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = waypoint['id']
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint['position'][0]
            marker.pose.position.y = waypoint['position'][1]
            marker.pose.position.z = 0.0  # Z position is zero for all waypoints
            marker.pose.orientation.w = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            self.marker_array.markers.append(marker)

            # Create the text marker for the waypoint name
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "waypoints"
            text_marker.id = waypoint['id'] + 1000  # Ensure unique ID for text marker
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = waypoint['position'][0]
            text_marker.pose.position.y = waypoint['position'][1]
            text_marker.pose.position.z = 0.2  # Slightly above the waypoint marker
            text_marker.pose.orientation.x = 0.0
            text_marker.pose.orientation.y = 0.0
            text_marker.pose.orientation.z = 0.0
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.1 
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 0.0
            text_marker.color.b = 0.0
            text_marker.text = f"{waypoint['name']} ({waypoint['id']})"

            self.marker_array.markers.append(text_marker)

            # Create the arrow marker for waypoint orientation indication
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "map"
            arrow_marker.header.stamp = rospy.Time.now()
            arrow_marker.ns = "waypoints"
            arrow_marker.id = waypoint['id'] + 2000  # Ensure unique ID for arrow marker
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.pose.position.x = waypoint['position'][0]
            arrow_marker.pose.position.y = waypoint['position'][1]
            arrow_marker.pose.position.z = 0.0  # Z position is zero for all waypoints
            arrow_marker.pose.orientation.w = waypoint['orientation'][0]
            arrow_marker.pose.orientation.x = waypoint['orientation'][1]
            arrow_marker.pose.orientation.y = waypoint['orientation'][2]
            arrow_marker.pose.orientation.z = waypoint['orientation'][3]
            arrow_marker.scale.x = 0.2  # Arrow length
            arrow_marker.scale.y = 0.05  # Arrow width
            arrow_marker.scale.z = 0.05  # Arrow height
            arrow_marker.color.a = 1.0
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 1.0
            arrow_marker.color.b = 0.0

            self.marker_array.markers.append(arrow_marker)

    
    def odom_callback(self, msg):
        if not self.use_apriltag:
            rospy.loginfo("Using sadkhfafh for localization")
            if not self.initial_pose_set:
                self.initial_duckie_marker_pose = self.duckiebot_marker.pose.position
                self.initial_duckie_marker_orientation = PyQuaternion(
                    self.duckiebot_marker.pose.orientation.w,
                    self.duckiebot_marker.pose.orientation.x,
                    self.duckiebot_marker.pose.orientation.y,
                    self.duckiebot_marker.pose.orientation.z
                )
                self.initial_odom_pose = msg.pose.pose.position
                self.initial_odom_orientation = PyQuaternion(
                    msg.pose.pose.orientation.w,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z
                )
                self.initial_pose_set = True

            # Calculate the relative change in position
            delta_translation = Point(
                msg.pose.pose.position.x - self.initial_odom_pose.x,
                msg.pose.pose.position.y - self.initial_odom_pose.y,
                msg.pose.pose.position.z - self.initial_odom_pose.z
            )

            # Calculate the relative change in orientation
            current_odom_orientation = PyQuaternion(
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z
            )
            delta_orientation = current_odom_orientation * self.initial_odom_orientation.inverse

            # Apply the relative change to the initial marker pose
            new_position = Point(
                self.initial_duckie_marker_pose.x + delta_translation.x,
                self.initial_duckie_marker_pose.y + delta_translation.y,
                self.initial_duckie_marker_pose.z + delta_translation.z
            )

            new_orientation = self.initial_duckie_marker_orientation * delta_orientation


            rospy.loginfo(f"New position: {new_position}")
            rospy.loginfo(f"New orientation: {new_orientation}")
            self.duckiebot_marker.pose.position = new_position
            self.duckiebot_marker.pose.orientation = Quaternion(
                new_orientation.x,
                new_orientation.y,
                new_orientation.z,
                new_orientation.w
            )


    def self_localization_callback(self, msg):
        closest_detection = None
        min_distance = float('inf')

        # find closest apriltag and use it for global localization
        for detection in msg.detections:
            position = detection.pose.pose.pose.position
            distance = np.sqrt(position.x**2 + position.y**2 + position.z**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_detection = detection


        if closest_detection:

            if min_distance < 0.7:
                angle = np.arctan2(closest_detection.pose.pose.pose.position.z, closest_detection.pose.pose.pose.position.x)
                if np.radians(60) < angle < np.radians(120):

                    self.use_apriltag = True
                    # get id of apriltag and compare it with global waypoint ids
                    tag_id = closest_detection.id[0]
                    
                    matching_waypoint = None
                    for waypoint in self.waypoints:
                        if waypoint['id'] == tag_id:
                            matching_waypoint = waypoint
                            break

                    # this part needs some comments, some stuff was only found by trial and error
                    if matching_waypoint:

                        camera_to_apriltag_transform = PyQuaternion(closest_detection.pose.pose.pose.orientation.w, closest_detection.pose.pose.pose.orientation.x, closest_detection.pose.pose.pose.orientation.y, closest_detection.pose.pose.pose.orientation.z)

                        # transform apriltag position with respect to real duckie to sim-duckie pos with respect to waypoint
                        relative_duckie_pos_sim = Point(
                            -closest_detection.pose.pose.pose.position.z,
                            closest_detection.pose.pose.pose.position.x,
                            0.0335
                        )

                        # get z axis rotation between waypoint and global coordinate system
                        waypoint_orientation = PyQuaternion(matching_waypoint['orientation'][3], matching_waypoint['orientation'][0], matching_waypoint['orientation'][1], matching_waypoint['orientation'][2])
                        global_map_orientation = PyQuaternion(1, 0, 0, 0)
                        yaw = (waypoint_orientation * global_map_orientation.inverse).yaw_pitch_roll[2]
                        yaw_quat = PyQuaternion(axis=[0, 0, 1], angle=yaw)


                        # rotate relative pos of sim-duckie to waypoint by its yaw in reference to the global coords.
                        rotated_rel_duckie_pos_sim_x = relative_duckie_pos_sim.x * np.cos(-yaw) - relative_duckie_pos_sim.y * np.sin(-yaw)
                        rotated_rel_duckie_pos_sim_y = relative_duckie_pos_sim.x * np.sin(-yaw) + relative_duckie_pos_sim.y * np.cos(-yaw)

                        # set global position of sim-duckie
                        self.duckiebot_marker.pose.position = Point(
                            matching_waypoint['position'][0] + rotated_rel_duckie_pos_sim_x,
                            matching_waypoint['position'][1] + rotated_rel_duckie_pos_sim_y,
                            0.0335
                        )

                        # set orientation of sim-duckie (qz and qy rotations due to different reference coord systems of duckie in sim vs real)
                        orientation = self.qz.inverse * self.qx.inverse * camera_to_apriltag_transform * self.camera_angle.inverse * yaw_quat * self.qz 

                        # set global orientation of sim-duckie
                        self.duckiebot_marker.pose.orientation = Quaternion(-orientation[1], -orientation[2], -orientation[3], orientation[0])
                        self.initial_pose_set = False  # Reset initial pose flag
                        return
                    
                    
        rospy.loginfo("Using odometry for localization")             
        self.use_apriltag = False

    def run(self):
        rate = rospy.Rate(10) 

        while not rospy.is_shutdown():
            
            self.marker_pub.publish(self.marker_array)
            self.duckiebot_pub.publish(self.duckiebot_marker)
            rate.sleep()

if __name__ == "__main__":
    try:
        node = VizNode()
        node.run()
    except rospy.ROSInterruptException:
        pass