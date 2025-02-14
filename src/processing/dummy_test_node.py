#!/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

class DummyPublisherNode(Node):
    def __init__(self):
        super().__init__('dummy_publisher_node')
        
        self.bridge = CvBridge()
        
        # Publishers
        self.front_view_pub = self.create_publisher(Image, '/oakd_front_view/color', 10)
        self.side_view_pub = self.create_publisher(Image, '/oakd_side_view/color', 10)
        self.wrist_view_pub = self.create_publisher(Image, '/oakd_wrist_view/color', 10)
        
        # Timer to publish messages periodically
        self.timer = self.create_timer(1.0, self.publish_images)
        
    def publish_images(self):
        # Create dummy images
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_image[:] = (255, 0, 0)  # Blue image
        
        # Convert OpenCV image to ROS Image message
        image_msg = self.bridge.cv2_to_imgmsg(dummy_image, encoding='bgr8')
        
        # Publish the images
        self.front_view_pub.publish(image_msg)
        self.side_view_pub.publish(image_msg)
        self.wrist_view_pub.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DummyPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()