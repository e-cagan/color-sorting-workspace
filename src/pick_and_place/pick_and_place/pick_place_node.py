"""
Module for pick and place node.
"""

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda as panda_robot

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sorting_interfaces.msg import DetectedObject
from sorting_interfaces.action import SortObject
from geometry_msgs.msg import Pose, Quaternion, Point


class PickPlaceNode(Node):
    """
    A node that receives action client requests and processes them on it's own action server.
    """

    def __init__(self):
        super().__init__('pick_place_node')

        # Callback group
        self.callback_group = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameter('planning_group', 'panda_arm')
        self.declare_parameter('end_effector_frame', 'panda_hand')
        self.declare_parameter('grasp_offset', 0.1)
        self.declare_parameter('place_offset', 0.1)
        self.declare_parameter('velocity_scaling', 0.5)
        self.declare_parameter('acceleration_scaling', 0.5)
        ## Bin postion Parameters (For now it's parameterized)
        self.declare_parameter('red_bin_pose', [0.5, 0.2, 0.0])
        self.declare_parameter('green_bin_pose', [0.5, 0.0, 0.0])
        self.declare_parameter('blue_bin_pose', [0.5, -0.2, 0.0])

        # Moveit2 instance
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda_robot.joint_names(),
            base_link_name=panda_robot.base_link_name(),
            end_effector_name=panda_robot.end_effector_name(),
            group_name=self.get_parameter('planning_group').value,
            callback_group=self.callback_group,
        )

        # Action server
        self.action_server = ActionServer(
            node=self,
            action_type=SortObject,
            action_name='sort_objects',
            execute_callback=self.execute_callback,
            callback_group=self.callback_group,
        )


    # Helper functions
    def get_bin_pose(self, color_label: str) -> Pose | None:
        """
        A helper function which converts bin's color label to corresponding pose
        """

        # Create the bin map to map color to values
        bin_map = {
            'red':   self.get_parameter('red_bin_pose').value,
            'green': self.get_parameter('green_bin_pose').value,
            'blue':  self.get_parameter('blue_bin_pose').value,
        }

        # Read the corresponding color from the bin map to access it's position (Point message type)
        point = bin_map.get(color_label)

        # Check the mapping for point of given color exists
        if point is None:
            return None

        # Create the Pose (Point + Quaternion) message to return after field filling
        pose = Pose()
        pose.position = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)   # identity

        return pose
    
    
    # Execute callback
    async def execute_callback(self, goal_handle):
        """
        A callback that executes the client request via action server.s
        """

        # Take goal based on goal handle request
        goal = goal_handle.request
        color_label = goal.color_label
        object_pose = goal.object_pose

        # Find the bin pose
        bin_pose = self.get_bin_pose(color_label)
        if bin_pose is None:
            goal_handle.abort()
            result = SortObject.Result()
            result.success = False
            result.message = f"Unknown color label: {color_label}"
            return result

        # TODO Sequence
        # ...

        # Return the result
        goal_handle.succeed()
        result = SortObject.Result()
        result.success = True
        result.message = f"Sorted {color_label} object successfully"
        return result


# Main function to simulate node lifecycle
def main(args=None):
    """
    Main function that handles node lifecycle.
    """

    # Node lifecycle with executor
    rclpy.init(args=args)
    node = PickPlaceNode()
    
    # Create the executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


# Call the main function to run node
if __name__ == "__main__":
    main()
