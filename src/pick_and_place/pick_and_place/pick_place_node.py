"""
Module for pick and place node.
"""

import copy
import asyncio
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
        ## Bin position parameters (parameterized for now, will be detected from camera later)
        self.declare_parameter('red_bin_pose', [0.5, 0.2, 0.0])
        self.declare_parameter('green_bin_pose', [0.5, 0.0, 0.0])
        self.declare_parameter('blue_bin_pose', [0.5, -0.2, 0.0])
        self.planning_group = self.get_parameter('planning_group').value
        self.end_effector_frame = self.get_parameter('end_effector_frame').value
        self.grasp_offset = self.get_parameter('grasp_offset').value
        self.place_offset = self.get_parameter('place_offset').value
        self.velocity_scaling = self.get_parameter('velocity_scaling').value
        self.acceleration_scaling = self.get_parameter('acceleration_scaling').value

        # MoveIt2 instances
        ## Arm instance — controls the 7-DOF panda arm group
        self.arm = MoveIt2(
            node=self,
            joint_names=panda_robot.joint_names(),
            base_link_name=panda_robot.base_link_name(),
            end_effector_name=panda_robot.end_effector_name(),
            group_name=self.planning_group,  # 'panda_arm'
            callback_group=self.callback_group,
        )

        ## Hand instance — controls the 2-finger gripper group
        self.hand = MoveIt2(
            node=self,
            joint_names=['panda_finger_joint1', 'panda_finger_joint2'],
            base_link_name=panda_robot.base_link_name(),
            end_effector_name=panda_robot.end_effector_name(),
            group_name=self.end_effector_frame,  # 'panda_hand'
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
        Converts a color label to its corresponding bin pose.
        Returns None if the color label is not recognized.
        """

        # Map each color label to its parameter value
        bin_map = {
            'red':   self.get_parameter('red_bin_pose').value,
            'green': self.get_parameter('green_bin_pose').value,
            'blue':  self.get_parameter('blue_bin_pose').value,
        }

        # Use .get() to avoid KeyError on unknown labels
        point = bin_map.get(color_label)
        if point is None:
            return None

        # Convert [x, y, z] list to Pose with identity orientation
        pose = Pose()
        pose.position = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # identity quaternion

        return pose


    def publish_feedback(self, goal_handle, status_msg: str) -> None:
        """
        Publishes a status string as action feedback to the client.
        """

        feedback = SortObject.Feedback()
        feedback.status = status_msg
        goal_handle.publish_feedback(feedback)
        self.get_logger().info(f"Feedback sent: {status_msg}")


    async def pre_grasp(self, object_pose: Pose) -> None:
        """
        Pre-grasp: Open gripper and move the arm to a safe position above the object.
        """

        pre_grasp_pose = copy.deepcopy(object_pose)
        pre_grasp_pose.position.z += self.grasp_offset

        # Open the gripper before approaching (0.04 m = fully open for Panda)
        self.hand.move_to_configuration([0.04, 0.04])
        self.hand.wait_until_executed()
        await asyncio.sleep(0)

        # Move the arm above the object
        self.arm.move_to_pose(
            position=[pre_grasp_pose.position.x, pre_grasp_pose.position.y, pre_grasp_pose.position.z],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0]
        )
        self.arm.wait_until_executed()
        await asyncio.sleep(0)


    async def grasp(self, object_pose: Pose) -> None:
        """
        Grasp: Descend to the object's position and close the gripper.
        Note: a small Z tolerance may be needed depending on object height
        to avoid colliding with the table surface.
        """

        grasp_pose = copy.deepcopy(object_pose)

        # Descend to the grasp position
        self.arm.move_to_pose(
            position=[grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0]
        )
        self.arm.wait_until_executed()
        await asyncio.sleep(0)

        # Close the fingers around the object
        # Not fully closed (0.0) to avoid crushing the object and destabilizing planning
        self.hand.move_to_configuration([0.01, 0.01])
        self.hand.wait_until_executed()
        await asyncio.sleep(0)

        # TODO: Add an Attach collision object call here to prevent the object
        # from slipping during transport in simulation


    async def post_grasp(self, object_pose: Pose) -> None:
        """
        Post-grasp: Lift the object back up to the pre-grasp height before transporting.
        """

        post_grasp_pose = copy.deepcopy(object_pose)
        post_grasp_pose.position.z += self.grasp_offset

        self.arm.move_to_pose(
            position=[post_grasp_pose.position.x, post_grasp_pose.position.y, post_grasp_pose.position.z],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0]
        )
        self.arm.wait_until_executed()
        await asyncio.sleep(0)


    async def pre_place(self, bin_pose: Pose) -> None:
        """
        Pre-place: Move the arm to a safe position above the target bin.
        """

        pre_place_pose = copy.deepcopy(bin_pose)
        pre_place_pose.position.z += self.place_offset

        self.arm.move_to_pose(
            position=[pre_place_pose.position.x, pre_place_pose.position.y, pre_place_pose.position.z],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0]
        )
        self.arm.wait_until_executed()
        await asyncio.sleep(0)


    async def place(self, bin_pose: Pose) -> None:
        """
        Place: Descend into the bin and open the gripper to release the object.
        """

        place_pose = copy.deepcopy(bin_pose)

        # Descend to the place position
        self.arm.move_to_pose(
            position=[place_pose.position.x, place_pose.position.y, place_pose.position.z],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0]
        )
        self.arm.wait_until_executed()
        await asyncio.sleep(0)

        # Open the gripper fully to drop the object into the bin
        self.hand.move_to_configuration([0.04, 0.04])
        self.hand.wait_until_executed()
        await asyncio.sleep(0)

        # TODO: Add a Detach collision object call here


    async def retreat(self, bin_pose: Pose) -> None:
        """
        Retreat: Lift up from the bin to a safe clearance height before the next cycle.
        """

        retreat_pose = copy.deepcopy(bin_pose)
        retreat_pose.position.z += self.place_offset + 0.1

        self.arm.move_to_pose(
            position=[retreat_pose.position.x, retreat_pose.position.y, retreat_pose.position.z],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0]
        )
        self.arm.wait_until_executed()
        await asyncio.sleep(0)


    # Execute callback
    async def execute_callback(self, goal_handle):
        """
        Main action server callback. Orchestrates the full pick-and-place sequence
        for a single detected object based on its color label and pose.
        """

        # Unpack goal fields
        goal = goal_handle.request
        color_label = goal.color_label
        object_pose = goal.object_pose

        # Resolve the target bin pose from the color label
        bin_pose = self.get_bin_pose(color_label)
        if bin_pose is None:
            goal_handle.abort()
            result = SortObject.Result()
            result.success = False
            result.message = f"Unknown color label: {color_label}"
            return result

        # Pick-and-place sequence
        self.publish_feedback(goal_handle, "moving_to_object")
        await self.pre_grasp(object_pose)

        self.publish_feedback(goal_handle, "grasping")
        await self.grasp(object_pose)

        self.publish_feedback(goal_handle, "moving_to_bin")
        await self.post_grasp(object_pose)
        await self.pre_place(bin_pose)

        self.publish_feedback(goal_handle, "placing")
        await self.place(bin_pose)
        await self.retreat(bin_pose)

        # Report success
        goal_handle.succeed()
        result = SortObject.Result()
        result.success = True
        result.message = f"Sorted {color_label} object successfully"
        return result


# Main function to handle node lifecycle
def main(args=None):
    """
    Initializes the node and spins with a MultiThreadedExecutor to allow
    concurrent action server execution alongside MoveIt2 callbacks.
    """

    rclpy.init(args=args)
    node = PickPlaceNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


# Entry point
if __name__ == "__main__":
    main()