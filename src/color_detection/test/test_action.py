# test_action.py
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sorting_interfaces.action import SortObject
from geometry_msgs.msg import Pose, Quaternion, Point

class TestActionClient(Node):
    def __init__(self):
        super().__init__('test_action_client')
        self._client = ActionClient(self, SortObject, 'sort_objects')

    def send_test_goal(self):
        self._client.wait_for_server()
        goal = SortObject.Goal()
        goal.color_label = 'red'
        goal.object_pose = Pose()
        goal.object_pose.position = Point(x=0.4, y=0.0, z=0.2)
        goal.object_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self._client.send_goal_async(goal)
        self.get_logger().info("Goal sent: red object at (0.4, 0.0, 0.2)")

def main():
    rclpy.init()
    node = TestActionClient()
    node.send_test_goal()
    rclpy.spin_once(node, timeout_sec=5.0)
    rclpy.shutdown()

if __name__ == '__main__':
    main()