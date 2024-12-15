#!/bin/env python3
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from utils.colorMasking import get_cropped_and_collor_maps
from utils.detect_cubes import detect_cubes

TOPICS_TO_SUBSCRIBE = {
    '/oakd_front_view/color': 'front',
    '/oakd_side_view/color': 'side',
    '/oakd_wrist_view/color': 'wrist'
}

class ColorProcessingNode(Node):
    def __init__(self):
        super().__init__('color_processing_node')
        
        self.COLOR_DETECTED = None
        self.START_TIME_FLAG = False
        self.START_TIME = 0
        
        self.bridge = CvBridge()
        
        # Get parameters
        self.declare_parameter('color_string', 'blue')
        
        # Subscribers with separate callbacks for each topic
        self.create_subscription(Image, '/oakd_front_view/color', self.front_image_callback, 20)
        self.create_subscription(Image, '/oakd_side_view/color', self.side_image_callback, 20)
        self.create_subscription(Image, '/oakd_wrist_view/color', self.wrist_image_callback, 20)
            
        # Publishers for each camera
        self.oakd_front_view_cropped_pub = self.create_publisher(Image, '/oakd_front_view/color_resized', 20)
        self.oakd_front_view_masked_pub = self.create_publisher(Image, '/oakd_front_view/color_masked', 20)
        self.oakd_front_view_grayed_pub = self.create_publisher(Image, '/oakd_front_view/color_grayed', 20)
        
        self.oakd_side_view_cropped_pub = self.create_publisher(Image, '/oakd_side_view/color_resized', 20)
        self.oakd_side_view_masked_pub = self.create_publisher(Image, '/oakd_side_view/color_masked', 20)
        self.oakd_side_view_grayed_pub = self.create_publisher(Image, '/oakd_side_view/color_grayed', 20)

        self.oakd_wrist_view_cropped_pub = self.create_publisher(Image, '/oakd_wrist_view/color_resized', 20)
        self.oakd_wrist_view_masked_pub = self.create_publisher(Image, '/oakd_wrist_view/color_masked', 20)
        self.oakd_wrist_view_grayed_pub = self.create_publisher(Image, '/oakd_wrist_view/color_grayed', 20)
        
        self.color_string_pub = self.create_publisher(String, '/color_string', 20)        

        
    def front_image_callback(self, msg):
        self.process_image_and_publish(msg, 'front')

    def side_image_callback(self, msg):
        self.process_image_and_publish(msg, 'side')

    def wrist_image_callback(self, msg):
        self.process_image_and_publish(msg, 'wrist')

    def process_image_and_publish(self, msg, camera):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Process the image
        image_cropped_resized, masks_combined_resized, gray_image_masked_resized, color_string = self.process_image(cv_image, camera)
        
        # Convert OpenCV images back to ROS Image messages
        image_cropped_resized_msg = self.bridge.cv2_to_imgmsg(image_cropped_resized, encoding='bgr8')
        masks_combined_resized_msg = self.bridge.cv2_to_imgmsg(masks_combined_resized, encoding='bgr8')
        gray_image_masked_resized_msg = self.bridge.cv2_to_imgmsg(gray_image_masked_resized, encoding='bgr8')
        
        # Publish the processed images and color string based on camera
        if camera == 'front':
            self.oakd_front_view_cropped_pub.publish(image_cropped_resized_msg)
            self.oakd_front_view_masked_pub.publish(masks_combined_resized_msg)
            self.oakd_front_view_grayed_pub.publish(gray_image_masked_resized_msg)
        elif camera == 'side':
            self.oakd_side_view_cropped_pub.publish(image_cropped_resized_msg)
            self.oakd_side_view_masked_pub.publish(masks_combined_resized_msg)
            self.oakd_side_view_grayed_pub.publish(gray_image_masked_resized_msg)
        elif camera == 'wrist':
            self.oakd_wrist_view_cropped_pub.publish(image_cropped_resized_msg)
            self.oakd_wrist_view_masked_pub.publish(masks_combined_resized_msg)
            self.oakd_wrist_view_grayed_pub.publish(gray_image_masked_resized_msg)
        
        print("Published images for", camera)
        # Publish color string
        # color_string_msg = String()
        # color_string_msg.data = color_string
        # self.color_string_pub.publish(color_string_msg)
        
    def search_for_color(self, image, camera):
        
        # Add to skip wrist camera 

        color_found = detect_cubes(image, camera, output_dir=None)

        print("Color found:", color_found)

        if color_found == None:
            return
        else :
            color_string = color_found
            if self.COLOR_DETECTED !=  color_string:
                if not self.START_TIME_FLAG:
                    self.START_TIME = time.time()
                    self.START_TIME_FLAG = True
                else:
                    if time.time() - self.START_TIME > 3:
                        self.COLOR_DETECTED = color_string
                        self.START_TIME_FLAG = False
            else:
                return

    def process_image(self, cv_image, camera):  
        image_bgr = cv_image.copy()

        # Write algo for finding dominant color in the image
        if camera != 'wrist':
            self.search_for_color(image_bgr, camera)

        color_string = self.COLOR_DETECTED

        if color_string == None:        
            image_cropped, masks_combined = get_cropped_and_collor_maps(image_bgr, camera, color_detected = None , output_dir=None)
            gray_image_masked = np.zeros_like(image_cropped)
        else: 
            image_cropped, masks_combined, gray_image_masked = get_cropped_and_collor_maps(image_bgr, camera, color_string, output_dir=None)

        image_cropped_resized  = cv2.resize(image_cropped, (224, 224))
        masks_combined_resized  = cv2.resize(masks_combined, (224, 224))
        gray_image_masked_resized  = cv2.resize(gray_image_masked, (224, 224))

        return image_cropped_resized, masks_combined_resized, gray_image_masked_resized, color_string

def main(args=None):
    rclpy.init(args=args)
    node = ColorProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
