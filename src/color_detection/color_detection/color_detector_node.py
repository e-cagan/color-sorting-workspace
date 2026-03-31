"""
Module for color detector node.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient

from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from sorting_interfaces.msg import DetectedObject
from sorting_interfaces.action import SortObject
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Point, PointStamped


# COLOR RANGES
COLOR_RANGES = {
    "red":   [([0,100,100], [10,255,255]), ([160,100,100], [180,255,255])],
    "green": [([40,50,50],  [80,255,255])],
    "blue":  [([100,50,50], [130,255,255])],
}


class ColorDetectorNode(Node):
    """
    A node that detects colors on HSV color space.
    """

    def __init__(self):
        super().__init__('color_detector_node')

        # Parameters and values
        self.declare_parameter('rgb_topic', '/camera/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('pixel_threshold', 400)
        self.declare_parameter('depth_scale', 1.0)
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.pixel_threshold = self.get_parameter('pixel_threshold').value
        self.depth_scale = self.get_parameter('depth_scale').value

        # Action client
        self.sort_action_cli = ActionClient(self, SortObject, 'sort_objects')

        # Publishers
        self.det_obj_pub = self.create_publisher(DetectedObject, '/detected_object', 10)

        # Subscribers
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.cam_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.cam_info_callback, 10)

        # Syncronize the subscribers to avoid conflicts and register the callback
        self.ts = ApproximateTimeSynchronizer(fs=[self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(cb=self.image_callback)

        # Other variables
        self.tracked_positions = list()
        self.cv_bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_transforms = TransformListener(buffer=self.tf_buffer, node=self)
        self.cam_matrix = None
        self.dist_coeffs = None

        self.get_logger().info("Color detector node started.")


    # Camera info callback
    def cam_info_callback(self, msg):
        """
        A callback that saves camera matrix and distortion coefficients.

        msg -> CamInfo
        """

        # Convert cam matrix to 3x3 array
        self.cam_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    
    # Image callback
    def image_callback(self, rgb_msg, depth_msg):
        """
        A callback that processes image and determines the detection.
        """

        # Check the camera matrix, distortion coefficients exists
        if self.cam_matrix is None:
            return
        if self.dist_coeffs is None:
            return
        
        # Convert rgb and depth messages to opencv frame (BGR format for RGB frame)
        rgb_frame = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_frame = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        # Convert BGR image to HSV space
        hsv_frame = cv2.cvtColor(src=rgb_frame, code=cv2.COLOR_BGR2HSV)

        # Take the detections
        rgb_detections = self.detect_color(hsv_frame)


    

    # Color detection pipeline
    def detect_color(self, hsv_img) -> list[tuple[str, tuple, float]]:
        """
        Returns list of (color_label, centroid_2d, confidence) for all detected objects.
        centroid_2d: (u, v) pixel coordinates
        confidence: normalized contour area [0.0 - 1.0]
        """
        
        # Create a list for storing detections
        detections = list()

        # Iterate trough colors
        for color, ranges in COLOR_RANGES.items():
            # Create an empty mask matrix to mask out the image
            combined_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
            
            # Create mask for colors
            for lower, upper in ranges:
                range_mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
                combined_mask = cv2.bitwise_or(combined_mask, range_mask)

            # Reduce the noise
            ## Define the kernel to use on morphological operations
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
            
            ## First apply opening to reduce little noises
            opened = cv2.morphologyEx(src=combined_mask, op=cv2.MORPH_OPEN, kernel=kernel)
            
            ## Then apply closing to fill up the possible holes inside of the mask
            closed = cv2.morphologyEx(src=opened, op=cv2.MORPH_CLOSE, kernel=kernel)

            # Find the contours of mask and iterate trough it
            contours, hierarchy = cv2.findContours(image=closed, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Find the area of the contour
                area = cv2.contourArea(contour=contour)

                # Ignore if the area is below pixel threshold
                if area < self.pixel_threshold:
                    continue
                
                # Find the hull area of the contour to calculate confidence score
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)

                # Calculate moments of the contour
                M = cv2.moments(contour)

                # Calculate horizontal (u) and vertical (v) pixel centers
                try:
                    u = M['m10'] / M['m00']
                    v = M['m01'] / M['m00']
                except ZeroDivisionError:
                    continue
                
                # Calculate confidence score
                ## Metric 1: Solidity [0.0 - 1.0]
                solidity = float(area) / hull_area if hull_area > 0 else 0.0
                
                ## Metric 2: Aspect Ratio Score [0.0 - 1.0]
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / float(h) if h > 0 else 0.0
                ## Aspect ratio 1'e ne kadar yakınsa skor o kadar yüksek olur.
                aspect_score = min(aspect_ratio, 1.0 / aspect_ratio) if aspect_ratio > 0 else 0.0
                
                # Weighted Sum
                ## Solidity is more trustworthy then aspect
                w_solidity = 0.7
                w_aspect = 0.3
                
                confidence = (w_solidity * solidity) + (w_aspect * aspect_score)
                confidence = float(np.clip(confidence, 0.0, 1.0))

                # Add detection to detections list
                detections.append((color, (u, v), confidence))

        return detections


    def get_3d_position(self, centroid_2d: tuple, depth_frame: np.ndarray) -> PointStamped | None:
        """
        A function which gets the 3D position of a point.
        """

        # Take fx, fy, cx and cy from camera matrix and take u and v to calculate X and Y for transformations
        fx, fy = self.cam_matrix[0][0], self.cam_matrix[1][1]
        cx, cy = self.cam_matrix[0][2], self.cam_matrix[1][2]
        u, v = centroid_2d

        # Calculate Z and scale Z. Also check if it's valid
        Z = depth_frame[int(v), int(u)] * self.depth_scale
        if Z <= 0 or np.isnan(Z):
            return None

        # Calculate X and Y
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # Create PointStamped message to conduct transformation between frames
        point_stamped = PointStamped()
        point_stamped.header.frame_id = self.camera_frame
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.point.x = X
        point_stamped.point.y = Y
        point_stamped.point.z = Z

        # Transform between frames
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame='base_link', source_frame=self.camera_frame,
                time=Time(), timeout=Duration(seconds=0.1)
            )
            point_in_base = do_transform_point(point=point_stamped, transform=transform)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None
        
        return point_in_base


    def is_duplicate(self, position):
        """
        A function which checks the detection is duplicate to avoid redundancy.
        """

        pass


# Main function to simulate node lifecycle
def main(args=None):
    """
    Main function that handles node lifecycle.
    """

    # Node lifecycle
    rclpy.init(args=args)
    node = ColorDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# Call the main function to run node
if __name__ == "__main__":
    main()
