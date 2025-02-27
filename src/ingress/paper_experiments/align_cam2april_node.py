#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
from pupil_apriltags import Detector
import math
from message_filters import Subscriber, TimeSynchronizer

class TagProcessorNode(Node):
    def __init__(self):
        super().__init__('tag_processor_node')
        
        # Initialize CvBridge to convert ROS Image messages to OpenCV format
        self.bridge = CvBridge()

        # Initialize AprilTag detector
        families = 'tag36h11'
        nthreads = 1
        quad_decimate = 1
        quad_sigma = 0
        refine_edges = 1
        decode_sharpening = 1
        debug = 0

        self.at_detector = Detector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            debug=debug,
        )

        # Subscriber for camera info and image topics
        self.image_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.camera_info_sub = Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')

        # Synchronize image and camera info topic
        self.ts = TimeSynchronizer([self.image_sub, self.camera_info_sub], 10)
        self.ts.registerCallback(self.image_callback)

        # Publisher for processed image
        self.image_processed_pub = self.create_publisher(Image, '/image_processed', 10)

    def image_callback(self, image_msg, camera_info_msg):
        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Ensure the image is of type uint8
        frame_u8 = frame.astype(np.uint8)

        # Camera intrinsics
        camera_matrix = np.array(camera_info_msg.k).reshape(3, 3)
        dist_coeffs = np.array(camera_info_msg.d)

        camera_fx = camera_matrix[0, 0]  # Focal length along x-axis
        camera_fy = camera_matrix[1, 1]  # Focal length along y-axis
        camera_cx = camera_matrix[0, 2]  # Principal point x-coordinate
        camera_cy = camera_matrix[1, 2]  # Principal point y-coordinate

        # Detect AprilTags in the frame
        tags = self.at_detector.detect(frame, estimate_tag_pose=True, camera_params=(camera_fx, camera_fy, camera_cx, camera_cy), tag_size=0.22)

        # Create a copy of the frame to draw on
        frame_with_arrows = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Initialize variables for Y-axis directions
        x1 = x2 = x3 = None
        tilt1 = tilt2 = tilt3 = None

        # For each detected tag
        for tag in tags:
            tag_id = tag.tag_id
            center = tag.center
            corners = tag.corners
            pose_t = tag.pose_t
            R = tag.pose_R     

            # Assign the X-axis direction to x1, x2, or x3 based on tag ID
            if tag_id == 0:
                x1 = R[:, 0]  # X-axis direction of tag 0
            if tag_id == 1:
                x2 = R[:, 0]  # X-axis direction of tag 1
            if tag_id == 2:
                x3 = R[:, 0]  # X-axis direction of tag 2

            # Compute the tilt angle (between the X-axis and camera's Z-axis)
            tilt = self.calculate_tilt(R)

            # Draw arrows for orientation visualization
            start_point = (int(center[0]), int(center[1]))
            end_point_x = start_point + (R[:, 0][:2] * 100).astype(int)
            end_point_y = start_point + (R[:, 1][:2] * 100).astype(int)
            cv2.arrowedLine(frame_with_arrows, start_point, tuple(end_point_x), (60, 60, 60), thickness=2)
            cv2.arrowedLine(frame_with_arrows, start_point, tuple(end_point_y), (80, 80, 80), thickness=2)

            # Draw tag boundaries
            for i in range(4):
                corner_1 = (int(corners[i][0]), int(corners[i][1]))
                corner_2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                cv2.line(frame_with_arrows, corner_1, corner_2, (0, 255, 0), 2)

            # Display tilt and alignment status for each tag
            if tag_id == 0:
                tilt1 = tilt
            if tag_id == 1:
                tilt2 = tilt
            if tag_id == 2:
                tilt3 = tilt

            # Draw the tilt angle as text on the image
            tilt_color = (0, 255, 0) if np.abs(tilt - 90) < 2 else (0, 0, 255)  # green if tilt < 3 degrees, else red
            cv2.putText(frame_with_arrows, f"Tilt: {tilt}", (int(center[0]), int(center[1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tilt_color, 2)

        # Display angles for alignment (calculate angles between x1, x2, x3)
        if x1 is not None and x2 is not None and x3 is not None:
            angle_prox, angle_dist = self.calc_angle(x1, x2, x3)

            # Display the angles on the image
            cv2.putText(
                frame_with_arrows,
                text=f"Proximal Angle: {angle_prox}",
                org=(250, 250),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255) if angle_prox > 5 else (0, 255, 0),
                thickness=2,
            )

            cv2.putText(
                frame_with_arrows,
                text=f"Distal Angle: {angle_dist}",
                org=(250, 330),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255) if angle_dist > 5 else (0, 255, 0),
                thickness=2,
            )

        # Convert back to ROS Image message and publish
        image_processed_msg = self.bridge.cv2_to_imgmsg(frame_with_arrows, encoding='bgr8')
        self.image_processed_pub.publish(image_processed_msg)

    def calc_angle(self, x1, x2, x3):
        x1[:2] /= np.linalg.norm(x1[:2])
        x2[:2] /= np.linalg.norm(x2[:2])
        x3[:2] /= np.linalg.norm(x3[:2])

        dot_product_proximal = np.dot(x1[:2], x2[:2])
        dot_product_distal = np.dot(x2[:2], x3[:2])

        angle_proximal = round(math.degrees(math.acos(dot_product_proximal)), 2)
        angle_distal = round(math.degrees(math.acos(dot_product_distal)), 2)

        return angle_proximal, angle_distal

    def calculate_tilt(self, R):
        # Compute the tilt angle between the X-axis of the tag and the Z-axis of the camera
        tag_x_axis = R[:, 0]  # X-axis direction of the tag
        camera_z_axis = np.array([0, 0, 1])  # Z-axis direction of the camera (aligned with the camera's optical axis)
        
        # Calculate the dot product
        dot_product = np.dot(tag_x_axis, camera_z_axis)
        tilt_angle = math.degrees(math.acos(dot_product))  # Convert the angle from radians to degrees

        return round(tilt_angle, 2)

def main(args=None):
    rclpy.init(args=args)
    node = TagProcessorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
