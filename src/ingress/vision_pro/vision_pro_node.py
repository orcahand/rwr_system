#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from faive_system.src.common.utils import numpy_to_float32_multiarray   
import sys
import os

# Add the directory containing VisionProTeleop to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VisionProTeleop'))

from avp_stream.streamer import VisionProStreamer

class VisionProNode(Node):
    def __init__(self, debug=False):
        super().__init__("vision_pro_node")
        print("VisionProNode initialized")
        # start tracker
        self.declare_parameter("vision_pro/ip", "10.93.181.127")
        self.get_logger().info('VisionProNode has been started.')

        ip = self.get_parameter("vision_pro/ip").value

        try:
            self.streamer = VisionProStreamer(ip=ip, record=True)
        except Exception as e:
            pass
        self.get_logger().info('VisionProStreamer has been started.')
        ingress_period = 0.005  # Timer period in seconds
        self.timer = self.create_timer(ingress_period, self.timer_publish_cb)

        self.ingress_mano_pub = self.create_publisher(
            Float32MultiArray, "/ingress/mano", 10
        )
        self.ingress_wrist_pub = self.create_publisher(
            PoseStamped, "/ingress/wrist", 10
        )
        self.debug = debug

    def timer_publish_cb(self):
        
        r = self.streamer.latest
                
        while (r is None):
            if (not (wait_cnt % 100000)):
                print("waiting for hand tracker", wait_cnt//100000)
            wait_cnt+=1
            r = self.streamer.latest
       
        right_wrist = r['right_wrist']
        right_fingers = r['right_fingers'] # np.array of shape (25, 4, 4)
        keypoint_positions = right_fingers[:, :3, 3]
        right_wrist_position = right_wrist[:,:3, 3]  
        assert keypoint_positions.shape == (27, 3), f"Unexpected shape: {keypoint_positions.shape}"
        
        forearmwrist = keypoint_positions[25] 
        forearm = keypoint_positions[26]     
 
        keypoint_positions = np.delete(keypoint_positions, [0, 5, 10, 15, 20, 25, 26], axis=0)
        keypoint_positions = np.insert(keypoint_positions, 0, forearmwrist, axis=0)
        keypoint_positions = np.insert(keypoint_positions, 0, forearm, axis=0)

        assert keypoint_positions.shape == (22, 3), f"Unexpected shape: {keypoint_positions.shape}"
        #print(keypoint_positions)
        
        keypoint_positions_msg = numpy_to_float32_multiarray(keypoint_positions)
        self.ingress_mano_pub.publish(keypoint_positions_msg)
        
        pinch_distance = r["right_pinch_distance"]  
        self.get_logger().info(f"Pinch distance: {pinch_distance}")
        
        wrist_msg = PoseStamped()
        wrist_msg.header.frame_id = "coil"
        wrist_msg.header.stamp = self.get_clock().now().to_msg()

        wrist_msg.pose.position = Point(
            x=0.0, y=0.0, z=0.0
        )

        # Assign orientation using Quaternion
        wrist_msg.pose.orientation = Quaternion(
            x=0.0, y=0.0, z=0.0, w=0.0
        )

            # Publish the message
        self.ingress_wrist_pub.publish(wrist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionProNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
