"""
Mock camera publisher for testing color detection pipeline.
Publishes a synthetic BGR image with red, green, and blue objects,
a flat depth image, and a camera info message.
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time


class MockCameraPublisher(Node):

    def __init__(self):
        super().__init__('mock_camera_publisher')

        self.bridge = CvBridge()

        self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_rect_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        self.timer = self.create_timer(0.1, self.publish)
        self.get_logger().info("Mock camera publisher started.")


    def publish(self):
        now = self.get_clock().now().to_msg()

        # --- RGB image ---
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Red cylinder: left
        cv2.circle(img, (160, 240), 40, (0, 0, 255), -1)
        # Green cylinder: center
        cv2.circle(img, (320, 240), 40, (0, 255, 0), -1)
        # Blue cylinder: right
        cv2.circle(img, (480, 240), 40, (255, 0, 0), -1)

        rgb_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = 'camera_color_optical_frame'

        # --- Depth image ---
        # Sabit 0.5m depth — tüm pikseller aynı mesafede
        depth = np.full((480, 640), 0.15, dtype=np.float32)
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = 'camera_color_optical_frame'

        # --- Camera info ---
        info = CameraInfo()
        info.header.stamp = now
        info.header.frame_id = 'camera_color_optical_frame'
        info.width = 640
        info.height = 480
        # Standart pinhole — fx=fy=500, cx=320, cy=240
        info.k = [
            500.0, 0.0, 320.0,
            0.0, 500.0, 240.0,
            0.0, 0.0, 1.0
        ]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.distortion_model = 'plumb_bob'

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info)


def main(args=None):
    rclpy.init(args=args)
    node = MockCameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()