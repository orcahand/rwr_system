#!/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge

TOPICS_TO_SUBSCRIBE = {
    '/oakd_front_view/color': 'front',
    '/oakd_side_view/color': 'side',
    '/oakd_wrist_view/color': 'wrist'
}

class ColorProcessingNode(Node):
    def __init__(self):
        super().__init__('color_processing_node')
        
        self.bridge = CvBridge()
        
        # Get parameters
        self.declare_parameter('grayed_images_with_colored_mask', False)
        self.declare_parameter('color_string', 'blue')
        
        # Subscribers with separate callbacks for each topic
        self.create_subscription(Image, '/oakd_front_view/color', self.front_image_callback, 20)
        self.create_subscription(Image, '/oakd_side_view/color', self.side_image_callback, 20)
        self.create_subscription(Image, '/oakd_wrist_view/color', self.wrist_image_callback, 20)
            
        # Publishers for each camera
        self.oakd_front_view_masked_pub = self.create_publisher(Image, '/oakd_front_view/color_resized', 20)
        self.oakd_front_view_masked_pub = self.create_publisher(Image, '/oakd_front_view/color_masked', 20)
        self.oakd_front_view_grayed_pub = self.create_publisher(Image, '/oakd_front_view/color_grayed', 20)
        
        self.oakd_side_view_masked_pub = self.create_publisher(Image, '/oakd_side_view/color_resized', 20)
        self.oakd_side_view_masked_pub = self.create_publisher(Image, '/oakd_side_view/color_masked', 20)
        self.oakd_side_view_grayed_pub = self.create_publisher(Image, '/oakd_side_view/color_grayed', 20)

        self.oakd_wrist_view_masked_pub = self.create_publisher(Image, '/oakd_wrist_view/color_resized', 20)
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
        color_masked, color_grayed, color_string = self.process_image(cv_image)
        
        # Convert OpenCV images back to ROS Image messages
        color_masked_msg = self.bridge.cv2_to_imgmsg(color_masked, encoding='bgr8')
        color_grayed_msg = self.bridge.cv2_to_imgmsg(color_grayed, encoding='mono8')
        
        # Publish the processed images and color string based on camera
        if camera == 'front':
            self.oakd_front_view_masked_pub.publish(color_masked_msg)
            self.oakd_front_view_grayed_pub.publish(color_grayed_msg)
        elif camera == 'side':
            self.oakd_side_view_masked_pub.publish(color_masked_msg)
            self.oakd_side_view_grayed_pub.publish(color_grayed_msg)
        elif camera == 'wrist':
            self.oakd_wrist_view_masked_pub.publish(color_masked_msg)
            self.oakd_wrist_view_grayed_pub.publish(color_grayed_msg)
        
        # Publish color string
        color_string_msg = String()
        color_string_msg.data = color_string
        self.color_string_pub.publish(color_string_msg)
        
    def process_image(self, cv_image):  
        
        color_masked = cv_image.copy()
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        color_string = self.get_parameter('color_string').value
        
        # TODO: Implement color masking and graying out the image
        
        resized_image = cv2.resize(cv_image, (224, 224))
        
        return color_masked, gray_image, color_string

def main(args=None):
    rclpy.init(args=args)
    node = ColorProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
