import cv2
import depthai as dai
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np

class FramePublisher(Node):
    def __init__(self):
        super().__init__('oakd_frame_publisher')
        self.image_pub = self.create_publisher(Image, 'image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, 'camera_info', 10)
        self.bridge = CvBridge()

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

        # Connect to device and start pipeline
        self.device = dai.Device(pipeline)
        self.video = self.device.getOutputQueue(name="video", maxSize=1, blocking=False)

    def publish_frame(self):
        while rclpy.ok():
            videoIn = self.video.get()
            frame = videoIn.getData().reshape((videoIn.getHeight(), videoIn.getWidth()))
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)

            # Publish frame as ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="mono8")
            self.image_pub.publish(ros_image)

            # Publish CameraInfo message
            camera_info = CameraInfo()
            camera_info.header.stamp = self.get_clock().now().to_msg()
            self.camera_info_pub.publish(camera_info)

            cv2.imshow("video", frame)

            if cv2.waitKey(1) == ord('q'):
                rclpy.shutdown()
                break

def main(args=None):
    rclpy.init(args=args)
    node = FramePublisher()
    try:
        node.publish_frame()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()