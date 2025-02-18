import cv2
import depthai as dai
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from pupil_apriltags import Detector
import math

def calc_angle(y1, y2, y3):



    # camera_z_axis = np.array([0, 0, 1])

    # # Project the alignment vectors of both AprilTags into the z-plane defined by the camera's z-axis
    # alignment_vector_x1_projected = alignment_vector_x1 - np.dot(alignment_vector_x1, camera_z_axis) * camera_z_axis
    # alignment_vector_y1_projected = alignment_vector_y1 - np.dot(alignment_vector_y1, camera_z_axis) * camera_z_axis

    # alignment_vector_x2_projected = alignment_vector_x2 - np.dot(alignment_vector_x2, camera_z_axis) * camera_z_axis
    # alignment_vector_y2_projected = alignment_vector_y2 - np.dot(alignment_vector_y2, camera_z_axis) * camera_z_axis

    # Normalize the projected alignment vectors
    y1[:2] /= np.linalg.norm(y1[:2])
    y2[:2] /= np.linalg.norm(y2[:2])
    y3[:2] /= np.linalg.norm(y3[:2])

    # Calculate the dot product between the y-alignment vectors in the z-plane
    dot_product_proximal = np.dot(y1[:2], y2[:2])
    dot_product_distal = np.dot(y2[:2], y3[:2])


    # Calculate the angle in radians
    angle_proximal = round(math.degrees(math.acos(dot_product_proximal)),2)
    angle_distal = round(math.degrees(math.acos(dot_product_distal)),2)

    return angle_proximal, angle_distal

# Initialize the AprilTag detector
families = 'tag36h11'
nthreads = 1
quad_decimate = 1
quad_sigma = 0
refine_edges = 1
decode_sharpening = 1
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

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camMono = pipeline.create(dai.node.MonoCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camMono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camMono.setBoardSocket(dai.CameraBoardSocket.LEFT)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

camMono.setFps(120)
# Linking
camMono.out.link(xoutVideo.input)

# Initialize ROS
rospy.init_node('frame_recorder', anonymous=True)
image_pub = rospy.Publisher('image_raw', Image, queue_size=10)
camera_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10)  # Create CameraInfo publisher
bridge = CvBridge()

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    while not rospy.is_shutdown():
        videoIn = video.get()
        frame = videoIn.getData().reshape((videoIn.getHeight(), videoIn.getWidth()))
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)


        # Estimate the camera intrinsic matrix (fx, fy, cx, cy) if available
        camera_matrix = np.array([[809.761910, 0.000000, 655.722303],
                          [0.000000, 810.768610, 383.834919],
                          [0.000000, 0.000000, 1.000000]])
        
        dist_coeffs = np.array([0.035809, -0.095205, -0.000315, 0.001731, 0.000000], dtype=np.float32)

        camera_fx = camera_matrix[0, 0]  # Focal length along x-axis
        camera_fy = camera_matrix[1, 1]  # Focal length along y-axis
        camera_cx = camera_matrix[0, 2]  # Principal point x-coordinate
        camera_cy = camera_matrix[1, 2]  # Principal point y-coordinate

        tags = at_detector.detect(frame, estimate_tag_pose=True, camera_params=(camera_fx, camera_fy, camera_cx, camera_cy), tag_size=0.23)
        frame_with_arrow = frame.copy()
        frame_with_arrow = cv2.cvtColor(frame_with_arrow, cv2.COLOR_GRAY2BGR)


        # Define the text position
        text_positionX = (50, 50)
        text_positionY = (600, 50)
        text_spacing = 30

        for tag in tags:

           
            # Retrieve the tag's ID and corners
            tag_id = tag.tag_id

            if tag_id == 0:
                color = (255, 0, 0)  # Blue
            elif tag_id == 1:
                color = (0, 255, 255)  # Yellow
            elif tag_id == 2:
                color = (0, 0, 255)  # Red
           
            center = tag.center
            corners = tag.corners
            
            pose_t = tag.pose_t

            R = tag.pose_R

            if tag_id == 0:
                y1 = R[:, 1]  # Y-axis direction in world coordinates
            if tag_id == 1:
                y2 = R[:, 1]  # Y-axis direction in world coordinates
            if tag_id == 2:
                y3 = R[:, 1]  # Y-axis direction in world coordinates
                angle_prox, angle_dist = calc_angle(y1,y2,y3)

                cv2.putText(
                    frame_with_arrow,
                    text=f"Proximal Angle: {angle_prox}",
                    org=(250, 250),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255) if angle_prox > 5 else (0, 255, 0),
                    thickness=2,
                )

                cv2.putText(
                    frame_with_arrow,
                    text=f"Distal Angle: {angle_dist}",
                    org=(250, 330),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255) if angle_dist > 5 else (0, 255, 0),
                    thickness=2,
                )

            # Define the starting point for the arrow
            start_point = (int(center[0]), int(center[1]))
            # Calculate the end points for the arrows
            end_point_x = start_point + (R[:, 0][:2] * 100).astype(int)
            end_point_y = start_point + (R[:, 1][:2] * 100).astype(int)
            # Draw the arrows on the image
            cv2.arrowedLine(frame_with_arrow, start_point, tuple(end_point_x), (0, 0, 255), thickness=2)
            cv2.arrowedLine(frame_with_arrow, start_point, tuple(end_point_y), (0, 255, 0), thickness=2)
    

            corners = tag.corners

            corner_01 = (int(corners[0][0]), int(corners[0][1]))
            corner_02 = (int(corners[1][0]), int(corners[1][1]))
            corner_03 = (int(corners[2][0]), int(corners[2][1]))
            corner_04 = (int(corners[3][0]), int(corners[3][1]))

            cv2.line(frame_with_arrow, corner_01, corner_02, color, 2)
            cv2.line(frame_with_arrow, corner_02, corner_03, color, 2)
            cv2.line(frame_with_arrow, corner_03, corner_04, color, 2)
            cv2.line(frame_with_arrow, corner_04, corner_01, color, 2)

            # Calculate the pose manually using tag corners
            object_points = np.array([
                [-0.5, -0.5, 0],   # Top-left corner
                [0.5, -0.5, 0],    # Top-right corner
                [0.5, 0.5, 0],     # Bottom-right corner
                [-0.5, 0.5, 0]     # Bottom-left corner
            ], dtype=np.float32)


            _, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

            perpendicular_distance = abs(tvec[2])
            #print(perpendicular_distance)
            
            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            
       
            # Retrieve the translation vector
            translation_vector = tvec.flatten()

            # Get the camera's optical axis
            optical_axis = np.array([0, 0, 1])

            # Calculate the dot product between the translation vector and the optical axis
            dot_product = np.dot(translation_vector, optical_axis)

            #perpendicular_distance = abs(dot_product) * 2
            #distance = np.linalg.norm(tvec) * 2
            #print(f"{tag_id}: " + str(perpendicular_distance))

            
    
            angle_threshold = 5.0  # Adjust this value
    
            # Assuming you have the rotation matrix as `rotation_matrix` from the previous steps
            # Calculate the angles around X and Y axes

           

            angle_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            angle_y = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
            
            
            # Convert the angles to degrees
            angle_x_deg = np.degrees(angle_x)
            angle_y_deg = np.degrees(angle_y)

            # Adjust the X angle to be within the range of 0 to 360 degrees
            if angle_x_deg < -90:
                angle_x_deg += 180.0
            elif angle_x_deg > 90:
                angle_x_deg -= 180.0

            if np.abs(angle_x_deg) < angle_threshold:
                cv2.putText(frame_with_arrow, f"Tag {tag_id}: Y is Perpendicular", text_positionX, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                text_positionX = (text_positionX[0], text_positionX[1] + text_spacing)
            if np.abs(angle_y_deg) < angle_threshold:
                cv2.putText(frame_with_arrow, f"Tag {tag_id}: X is Perpendicular", text_positionY, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                text_positionY = (text_positionY[0], text_positionY[1] + text_spacing)
           
                
          


        cv2.imshow("tags", frame_with_arrow)

        if cv2.waitKey(1) == ord('q'):
            break