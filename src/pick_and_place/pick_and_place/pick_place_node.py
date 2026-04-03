"""
Module for pick and place node.
"""

import copy
import time
from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import panda as panda_robot

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.action.server import GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from control_msgs.action import GripperCommand
from sorting_interfaces.action import SortObject
from geometry_msgs.msg import Pose, Quaternion, Point
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import SetBool


class PickPlaceNode(Node):
    """
    Pick-and-place node using:
      - Finger gripper (GripperCommand) for the open/close animation
      - Vacuum gripper plugin (/vacuum_gripper/switch) for physical holding in Gazebo
      - MoveIt2 collision-object attach/detach for collision-aware planning
    """

    def __init__(self):
        super().__init__('pick_place_node')

        self.callback_group = MutuallyExclusiveCallbackGroup()

        # Parameters
        self.declare_parameter('planning_group', 'panda_arm')
        self.declare_parameter('end_effector_frame', 'panda_hand')
        self.declare_parameter('grasp_offset', 0.1)
        self.declare_parameter('place_offset', 0.1)
        self.declare_parameter('velocity_scaling', 0.5)
        self.declare_parameter('acceleration_scaling', 0.5)
        self.declare_parameter('red_bin_pose',   [0.5,  0.2, 0.0])
        self.declare_parameter('green_bin_pose', [0.5,  0.0, 0.0])
        self.declare_parameter('blue_bin_pose',  [0.5, -0.2, 0.0])

        self.planning_group = self.get_parameter('planning_group').value
        self.end_effector_frame = self.get_parameter('end_effector_frame').value
        self.grasp_offset = self.get_parameter('grasp_offset').value
        self.place_offset = self.get_parameter('place_offset').value
        self.velocity_scaling = self.get_parameter('velocity_scaling').value
        self.acceleration_scaling = self.get_parameter('acceleration_scaling').value

        self.arm = None  # initialized later

        # Finger gripper
        self.gripper_client = ActionClient(
            self, GripperCommand, '/panda_hand_controller/gripper_cmd',
            callback_group=self.callback_group)

        # Vacuum gripper service  (std_srvs/srv/SetBool)
        #   True  → suction on  (object sticks to panda_hand_tcp)
        #   False → suction off (object drops)
        self.vacuum_client = self.create_client(
            SetBool, '/vacuum_gripper/switch',
            callback_group=self.callback_group)

        # MoveIt2 planning-scene publishers
        self.attached_obj_pub = self.create_publisher(
            AttachedCollisionObject, '/attached_collision_object', 10)
        self.collision_obj_pub = self.create_publisher(
            CollisionObject, '/collision_object', 10)

        # Action server
        self.action_server = ActionServer(
            node=self, action_type=SortObject, action_name='sort_objects',
            execute_callback=self.execute_callback,
            callback_group=self.callback_group,
            goal_callback=lambda goal: GoalResponse.ACCEPT)

        self.get_logger().info("Pick place node started.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_bin_pose(self, color_label: str) -> Pose | None:
        bin_map = {
            'red':   self.get_parameter('red_bin_pose').value,
            'green': self.get_parameter('green_bin_pose').value,
            'blue':  self.get_parameter('blue_bin_pose').value,
        }
        point = bin_map.get(color_label)
        if point is None:
            return None
        pose = Pose()
        pose.position    = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return pose

    def init_arm(self, executor):
        self.arm = MoveIt2(
            node=self,
            joint_names=panda_robot.joint_names(),
            base_link_name=panda_robot.base_link_name(),
            end_effector_name='panda_hand',
            group_name=self.planning_group,
            callback_group=self.callback_group,
            execute_via_moveit=True,
        )
        self.arm.executor = executor
        self.get_logger().info("Arm initialized.")

    def publish_feedback(self, goal_handle, status_msg: str) -> None:
        feedback = SortObject.Feedback()
        feedback.status = status_msg
        goal_handle.publish_feedback(feedback)
        self.get_logger().info(f"Feedback sent: {status_msg}")

    def _call_vacuum(self, on: bool) -> None:
        """Call /vacuum_gripper/switch synchronously (True=on, False=off)."""
        if not self.vacuum_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Vacuum gripper service not available — skipping")
            return
        req = SetBool.Request()
        req.data = on
        future = self.vacuum_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is not None:
            self.get_logger().info(
                f"Vacuum {'ON' if on else 'OFF'}: {future.result().message}")
        else:
            self.get_logger().warn("Vacuum service call timed out")

    def attach_object(self, object_pose: Pose, object_id: str = 'grasped_object') -> None:
        """Attach cylinder to panda_hand in MoveIt planning scene."""
        obj = AttachedCollisionObject()
        obj.link_name = 'panda_hand'
        obj.object.id = object_id
        obj.object.header.frame_id = 'panda_link0'
        obj.object.header.stamp = self.get_clock().now().to_msg()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = [0.03, 0.02]
        obj.object.primitives = [primitive]
        obj.object.primitive_poses = [object_pose]
        obj.object.operation = CollisionObject.ADD
        obj.touch_links = [
            'panda_hand', 'panda_leftfinger', 'panda_rightfinger',
            'panda_hand_tcp', 'panda_hand_sc',
            'panda_link6', 'panda_link7', 'panda_link8',
        ]
        self.attached_obj_pub.publish(obj)
        self.get_logger().info(f"MoveIt: attached '{object_id}'")

    def detach_object(self, object_id: str = 'grasped_object') -> None:
        """Remove attached collision object from MoveIt planning scene."""
        detach = AttachedCollisionObject()
        detach.link_name = 'panda_hand'
        detach.object.id = object_id
        detach.object.operation = CollisionObject.REMOVE
        self.attached_obj_pub.publish(detach)
        remove = CollisionObject()
        remove.id = object_id
        remove.operation = CollisionObject.REMOVE
        remove.header.frame_id = 'panda_link0'
        remove.header.stamp = self.get_clock().now().to_msg()
        self.collision_obj_pub.publish(remove)
        self.get_logger().info(f"MoveIt: detached '{object_id}'")

    # ── motion primitives ─────────────────────────────────────────────────────

    def _move_and_wait(self, position, quat_xyzw=[1.0, 0.0, 0.0, 0.0]):
        self.arm.move_to_pose(position=position, quat_xyzw=quat_xyzw)
        while self.arm.query_state() != MoveIt2State.IDLE:
            time.sleep(0.1)

    async def pre_grasp(self, object_pose: Pose) -> None:
        pre = copy.deepcopy(object_pose)
        pre.position.z += self.grasp_offset
        await self.move_finger_gripper(0.04)
        self._move_and_wait([pre.position.x, pre.position.y, pre.position.z])

    async def grasp(self, object_pose: Pose) -> None:
        p = object_pose.position
        self._move_and_wait([p.x, p.y, p.z])
        await self.move_finger_gripper(0.01)   # close fingers
        self._call_vacuum(True)                 # vacuum ON
        self.attach_object(object_pose)

    async def post_grasp(self, object_pose: Pose) -> None:
        post = copy.deepcopy(object_pose)
        post.position.z += self.grasp_offset
        self._move_and_wait([post.position.x, post.position.y, post.position.z])

    async def pre_place(self, bin_pose: Pose) -> None:
        pre = copy.deepcopy(bin_pose)
        pre.position.z += self.place_offset
        self._move_and_wait([pre.position.x, pre.position.y, pre.position.z])

    async def place(self, bin_pose: Pose) -> None:
        p = bin_pose.position
        self._move_and_wait([p.x, p.y, p.z])
        self._call_vacuum(False)               # vacuum OFF → object drops
        await self.move_finger_gripper(0.04)   # open fingers
        self.detach_object()

    async def retreat(self, bin_pose: Pose) -> None:
        retreat = copy.deepcopy(bin_pose)
        retreat.position.z += self.place_offset + 0.1
        self._move_and_wait([retreat.position.x, retreat.position.y, retreat.position.z])

    async def move_finger_gripper(self, position: float) -> None:
        goal = GripperCommand.Goal()
        goal.command.position   = position
        goal.command.max_effort = 50.0
        if not self.gripper_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Finger gripper not available")
            return
        goal_handle = await self.gripper_client.send_goal_async(goal)
        if not goal_handle.accepted:
            self.get_logger().warn("Finger gripper goal rejected")
            return
        await goal_handle.get_result_async()

    # ── action server ─────────────────────────────────────────────────────────

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request
        color_label = goal.color_label
        object_pose = goal.object_pose

        bin_pose = self.get_bin_pose(color_label)
        if bin_pose is None:
            goal_handle.abort()
            result = SortObject.Result()
            result.success = False
            result.message = f"Unknown color label: {color_label}"
            return result

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

        goal_handle.succeed()
        result = SortObject.Result()
        result.success = True
        result.message = f"Sorted {color_label} object successfully"
        return result


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    node.init_arm(executor)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()