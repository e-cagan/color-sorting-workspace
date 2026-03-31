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
