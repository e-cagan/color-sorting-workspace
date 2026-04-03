"""
Module for sorting launch file.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    Function that generates launch description.
    """

    config_dir = os.path.join(
        get_package_share_directory('sorting_bringup'), 'config'
    )

    color_detector_node = Node(
        package='color_detection',
        executable='color_detector_node',
        name='color_detector_node',
        output='screen',
        parameters=[os.path.join(config_dir, 'color_detector.yaml')],
    )

    pick_place_node = Node(
        package='pick_and_place',
        executable='pick_place_node',
        name='pick_place_node',
        output='screen',
        parameters=[os.path.join(config_dir, 'pick_place.yaml')],
    )

    return LaunchDescription([
        TimerAction(period=30.0, actions=[color_detector_node]),
        TimerAction(period=30.0, actions=[pick_place_node]),
    ])