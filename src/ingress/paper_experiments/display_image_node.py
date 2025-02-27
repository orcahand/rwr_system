#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DisplayImageNode(Node):
    def __init__(self):
        super().__init__('display_image_node')
        
        # Declare the image topic parameter (default '/image_processed')
        self.declare_parameter('image_topic', '/image_processed')

        # Get the image topic from parameters
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        # Create a CvBridge object to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,
            image_topic,  # Using the topic from the parameter
            self.image_callback,  # Callback function when a new message is received
            10  # QoS (Quality of Service) setting, adjust based on needs
        )
        self.subscription  # Prevent unused variable warning

    def image_callback(self, msg):
        try:
            # Convert the ROS image message to OpenCV format (Grayscale)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Display the image in a window using OpenCV
            cv2.imshow("Processed Image", frame)

            # Wait for a key press to handle events (e.g., closing the window)
            cv2.waitKey(1)  # 1ms delay to refresh the window

        except Exception as e:
            self.get_logger().error(f"Error converting ROS image to OpenCV format: {e}")

def main(args=None):
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)

    # Create and spin the image display node
    node = DisplayImageNode()
    rclpy.spin(node)

    # Shutdown ROS2 when the node is stopped
    node.destroy_node()
    rclpy.shutdown()

    # Close OpenCV windows on shutdown
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
