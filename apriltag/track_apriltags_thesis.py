# !/usr/bin/env python3

"""Track bounding boxes within video.

Usage: python3 track_markers.py -fv <filepath_video>  -fm <filepath_intr_matrix> -n <num_tags> -s <start_frame> -e <end_frame> -d <depth>

Example: python3 track_markers.py -fv ~/Downloads/tracking/vid.mp4 -fm /Downloads/intrinsic_matrix.csv -n 2 -s 10 -e 50

Script for tracking N manually chosen bounding boxes within video. Point this script to the video file you would like to track and choose how many N bounding boxes are desired. These bounding box centers (markers) are stored in a CSV file afterwards. Two videos are created as well, one is the original video cut to [start_frame, end_frame], and the other is with the tracking bounding boxes displayed. If you for some reason desire to quite the tracking earlier than end_frame, you can press q to exit out.

Note: OpenCV installation can sometimes have trouble with cv2.legacy.MultiTracker_create(), the version that worked is:
opencv-contrib-python 4.5.2.52
"""
import rosbag2_py
from pupil_apriltags import Detector
import click
import cv2
import numpy as np
import csv
import pandas as pd
import copy
import os
import datetime
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float32MultiArray




@click.command()
@click.option(
    "--filepath_data",
    "-f",
    help="Filepath of bag file that should be analyzed.",
    required=True,
)

@click.option(
    "--start_frame", "-s", default=0, help="Relevant starting frame of video."
)
@click.option(
    "--end_frame", "-e", type=int, help="Relevant ending frame of video."
)

def track_markers(filepath_data: str, start_frame: int, end_frame: int):

    num_tags = 3

    families = 'tag36h11'
    nthreads = 1
    quad_decimate = 1
    quad_sigma = 0
    refine_edges = 1
    decode_sharpening = 0.25
    debug = 0

    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    intr_matrix_data = []
    with open('intrinsic_matrices/intr_matrix_oak.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            intr_matrix_data.append(row)
    intr_matrix = np.array(intr_matrix_data, dtype=float)
    camera_fx = intr_matrix[0, 0]  # Focal length along x-axis
    camera_fy = intr_matrix[1, 1]  # Focal length along y-axis
    camera_cx = intr_matrix[0, 2]  # Principal point x-coordinate
    camera_cy = intr_matrix[1, 2]  # Principal point y-coordinate

    bag_file = filepath_data # Path to the bag file

    # Initialize rosbag2 reader
    storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    bridge = CvBridge()

    # Get the number of messages in the bag file
    total_messages = 0
    while reader.has_next():
        reader.read_next()
        total_messages += 1

    # Reinitialize the reader to iterate over the messages again
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Iterate over the messages in the bag file
    frames = []
    timestamps = []
    count = 0

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == '/camera/camera/color/image_raw':
            img_msg = deserialize_message(data, Image)
            frame = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            frames.append(frame)
            timestamps.append(t)
            # Update progress
            count += 1
            progress = count / total_messages * 100
            print(f"Progress: {progress:.2f}%")

    total_frames = len(frames)
    if not end_frame:
        end_frame = total_frames
    # Initialize variables
    framenum = 0

    coords = []
    framenum = 0
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255)]


    #########
    previous_tag_ids = {}

    for frame in frames:
        show_frame = copy.deepcopy(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(show_frame.shape)
        # show_frame = cv2.cvtColor(show_frame, cv2.COLOR_GRAY2RGB)

        framenum += 1
        if framenum < start_frame:
            continue
        if framenum > end_frame:
            break
        tags = at_detector.detect(gray_frame, estimate_tag_pose=True, camera_params=(camera_fx, camera_fy, camera_cx, camera_cy), tag_size=0.23)
        # tags = at_detector.detect(frame, estimate_tag_pose=True, camera_params=(camera_fx, camera_fy, camera_cx, camera_cy), tag_size=0.23)

        if len(tags) != num_tags:
            detected_tag_ids = []
            for tag in tags:
                tag_id = tag.tag_id
                detected_tag_ids.append(tag_id)
                
            # Check if tag_id 0 is missing
            if 0 not in detected_tag_ids:
                print("Tag 0 is missing in the frame!")
                old_tag = previous_tag_ids[0]
                tags.insert(0,old_tag)
            

            # Check if tag_id 1 is missing
            if 1 not in detected_tag_ids:
                print("Tag 1 is missing in the frame!")
                old_tag = previous_tag_ids[1]
                tags.insert(1,old_tag)

            # Check if tag_id 2 is missing
            if 2 not in detected_tag_ids:
                print("Tag 2 is missing in the frame!")
                old_tag = previous_tag_ids[2]
                tags.insert(2,old_tag)

            
        org_x = 100
        org_y = 70
        vertical_spacing = 30

        curr_coords = []
        
        for i, tag in enumerate(tags):  
            tag_id = tag.tag_id
            center = tag.center
            corners = tag.corners
            R = tag.pose_R

            # Extract the alignment vector
            alignment_vector_x = R[:, 0]  # X-axis direction in world coordinates
            alignment_vector_y = R[:, 1]  # Y-axis direction in world coordinates

            previous_tag_ids[tag_id] = tag

            center = (int(center[0]), int(center[1]))

            length_scale = 100

            # Calculate the end points for the arrows
            end_point_x = center + (alignment_vector_x[:2] * length_scale).astype(int)
            end_point_y = center + (alignment_vector_y[:2] * length_scale).astype(int)

            # Draw the arrows on the image
            cv2.arrowedLine(show_frame, center, tuple(end_point_x), (0, 0, 255), thickness=2)
            cv2.arrowedLine(show_frame, center, tuple(end_point_y), (0, 255, 0), thickness=2)
            corner_01 = (int(corners[0][0]), int(corners[0][1]))
            corner_02 = (int(corners[1][0]), int(corners[1][1]))
            corner_03 = (int(corners[2][0]), int(corners[2][1]))
            corner_04 = (int(corners[3][0]), int(corners[3][1]))
            cv2.circle(show_frame, (center[0], center[1]), 5, (0, 0, 255), 2)

            cv2.line(show_frame, (corner_01[0], corner_01[1]),
                    (corner_02[0], corner_02[1]), (255, 0, 0), 2)
            cv2.line(show_frame, (corner_02[0], corner_02[1]),
                    (corner_03[0], corner_03[1]), (255, 0, 0), 2)
            cv2.line(show_frame, (corner_03[0], corner_03[1]),
                    (corner_04[0], corner_04[1]), (0, 255, 0), 2)
            cv2.line(show_frame, (corner_04[0], corner_04[1]),
                    (corner_01[0], corner_01[1]), (0, 255, 0), 2)
            cv2.putText(show_frame, str(tag_id), (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[i], 2, cv2.LINE_AA)

            new_y = org_y + i * vertical_spacing

            cv2.putText(
                show_frame,
                text=f"Tag {tag_id}: " + str(alignment_vector_y),
                org=(org_x, new_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=colors[i],
                thickness=2,
            )
            
            cv2.putText(
                show_frame,
                text=f"{framenum}/{total_frames} Frames",
                org=(20, frame.shape[0]-20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(0, 0, 255),
                thickness=2,
             )
            

            curr_coords.append(np.array(alignment_vector_y))


        coords.append(np.array(curr_coords).flatten())


        cv2.imshow(f"Tracker (press Q to exit early)", show_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exited early!")
            break
    
    cv2.destroyAllWindows()


    formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    df_coords = pd.DataFrame(coords)  
    column_headers = []
    for i in range(3):
        for coord in ['X', 'Y', 'Z']:
            column_header = f"Tag Alignment Vector {i+1}: {coord}-Coord"
            column_headers.append(column_header)
    df_coords.columns = column_headers
    df_coords['Timestamp'] = timestamps  
    df_coords.to_csv("data_files/groundtruth_marker_positions.csv", index=False)
    df_coords.to_csv(f"data_log/groundtruth_marker_positions_{formatted_timestamp}.csv", index=False)

    print("Ground-Truth Marker Positions saved to data_files/groundtruth_marker_positions.csv successfully.")

    print(timestamps)
    if len(timestamps) > 1:
        time_diffs = np.diff(timestamps)
        avg_time_diff = np.mean(time_diffs)
        frequency = 1.0 / avg_time_diff
        print(f"Frequency of the signal: {frequency:.2f} Hz")
    else:
        print("Not enough timestamps to calculate frequency")

    joint_commands_topic = "/hand/policy_output"

    # Reset reader to read joint commands
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    command_data = {
        "Timestamp": [],
        "root2index_pp": [],
        "index_pp2mp": []
    }

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == joint_commands_topic:
            msg = deserialize_message(data, Float32MultiArray)
            command_data["Timestamp"].append(t)
            command_data["root2index_pp"].append(msg.data[6])
            command_data["index_pp2mp"].append(msg.data[7])

    command_df = pd.DataFrame(command_data)

    command_df.to_csv("data_files/commanded_joint_states.csv", index=False)
    command_df.to_csv(f"data_log/commanded_joint_states_{formatted_timestamp}.csv", index=False)
    print("Proprioception Angles saved to data_files/joint_states.csv successfully.")


if __name__ == "__main__":
    track_markers()