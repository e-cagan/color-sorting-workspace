"""
Module for color detector node.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.task import Future
from rclpy.duration import Duration
from rclpy.action import ActionClient

from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from sorting_interfaces.msg import DetectedObject
from sorting_interfaces.action import SortObject
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Point, PointStamped, Pose, Quaternion


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
        self.declare_parameter('dedup_threshold', 0.05)
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('depth_scale', 1.0)
        self.declare_parameter('base_frame', 'panda_link0')
        self.base_frame = self.get_parameter('base_frame').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.pixel_threshold = self.get_parameter('pixel_threshold').value
        self.dedup_threshold = self.get_parameter('dedup_threshold').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.min_confidence = self.get_parameter('min_confidence').value

        # Action client
        self.sort_action_cli = ActionClient(self, SortObject, 'sort_objects')

        # Publishers
        self.det_obj_pub = self.create_publisher(DetectedObject, '/detected_object', 10)

        # Subscribers
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.cam_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.cam_info_callback, 10)

        # Syncronize the subscribers to avoid conflicts and register the callback
        self.ts = ApproximateTimeSynchronizer(fs=[self.rgb_sub, self.depth_sub], queue_size=10, slop=2.0)
        self.ts.registerCallback(cb=self.image_callback)

        # Other variables
        self.tracked_positions = list()
        self.cv_bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_transforms = TransformListener(buffer=self.tf_buffer, node=self)
        self.last_goal_time = self.get_clock().now()
        self.goal_cooldown_sec = 30.0
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
        self.get_logger().info("Camera info received!")

    
    # Send goal function
    def send_goal(self, color_label: str, object_pose: Point) -> Future | None:
        """
        A function that sends goal to action server.
        """
        
        goal_msg = SortObject.Goal()
        
        # Convert object pose to an actual Pose (Point + Quaternion) instead of Point
        pose = Pose()
        pose.position = object_pose
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # identity
        goal_msg.color_label = color_label
        goal_msg.object_pose = pose

        if not self.sort_action_cli.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn("Action server not available, skipping.")
            return None
        
        future = self.sort_action_cli.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(
            lambda f, pos=object_pose: self.goal_response_callback(f, pos)
        )

        return future
    
    
    # Goal response callback
    def goal_response_callback(self, future: Future, position: Point) -> None:
        """
        A callback that takes goal response.
        """
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected")
            # Remove even if got rejected
            self.tracked_positions.remove(position)
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f, pos=position: self.result_callback(f, pos)
        )

    
    # Result callback
    def result_callback(self, future: Future, position: Point) -> None:
        """
        A callback that takes the result and removes the result in detections if result is in it.
        """
        
        result = future.result().result
        self.get_logger().info(f"Sort result: {result.message}")
        if position in self.tracked_positions:
            self.tracked_positions.remove(position)

    
    # Feedback callback
    def feedback_callback(self, feedback_msg) -> None:
        """
        A callback that logs the action feedback.
        """
        
        feedback = feedback_msg.feedback
        self.get_logger().debug(f"Feedback: {feedback.status}")


    # Image callback
    def image_callback(self, rgb_msg, depth_msg):
        """
        A callback that processes image and determines the detection.
        """
        self.get_logger().info("Image callback triggered!")
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
        hsv_detections = self.detect_color(hsv_frame)

        # Iterate trough detections
        for color, centroid_2d, confidence in hsv_detections:
            # Take the position of detection and check if it's none
            position = self.get_3d_position(centroid_2d=centroid_2d, depth_frame=depth_frame)
            if position is None:
                continue

            # Check if the confidence is lower than minimum confidence threshold
            if confidence < self.min_confidence:
                continue
                
            # Check the detection is duplicate
            if self.is_duplicate(position=position):
                continue
            self.tracked_positions.append(position)

            # Publish DetectedObject message for debug before sending goal
            debug_msg = DetectedObject()
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = 'base_link'
            debug_msg.color_label = color
            debug_msg.position = position
            debug_msg.confidence = confidence
            self.det_obj_pub.publish(debug_msg)

            # Add cooldown
            now = self.get_clock().now()
            elapsed = (now - self.last_goal_time).nanoseconds / 1e9
            if elapsed < self.goal_cooldown_sec:
                continue
            self.last_goal_time = now

            # Send goal to action
            self.send_goal(color_label=color, object_pose=position)

    
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


    # 3D position obtainer function
    def get_3d_position(self, centroid_2d: tuple, depth_frame: np.ndarray) -> Point | None:
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
                target_frame=self.base_frame, source_frame=self.camera_frame,
                time=Time(), timeout=Duration(seconds=0.1)
            )
            point_in_base = do_transform_point(point=point_stamped, transform=transform)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None
        
        return point_in_base.point


    # Is duplicate function
    def is_duplicate(self, position: Point) -> bool:
        """
        A function which checks the detection is duplicate to avoid redundancy.
        """

        # Iterate trough the tracked positions to estimate distance
        for tracked in self.tracked_positions:
            # Calculate eucladian distance and check if the distance is less than or equal to dedup threshold
            distance = np.sqrt((position.x - tracked.x)**2 + (position.y - tracked.y)**2 + (position.z - tracked.z)**2)
            if distance <= self.dedup_threshold:
                return True
            
        return False


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
