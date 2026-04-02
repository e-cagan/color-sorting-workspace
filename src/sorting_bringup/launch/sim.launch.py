"""
Module for simulation launch file.
"""

import os
import subprocess
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """
    Launches Gazebo simulation with Panda robot, MoveIt2, and RViz.
    robot_description is processed via xacro subprocess to preserve
    ros2_control tags that MoveItConfigsBuilder would otherwise strip.
    """

    bringup_dir = get_package_share_directory('sorting_bringup')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    # Process URDF via xacro directly — preserves ros2_control tags
    xacro_file = os.path.join(bringup_dir, 'urdf', 'panda_with_camera.urdf.xacro')
    urdf_str = subprocess.check_output(['xacro', xacro_file]).decode('utf-8')

    # MoveIt2 config — uses same URDF for planning/kinematics
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path=xacro_file,
        )
        .robot_description_semantic(
            file_path=os.path.join(bringup_dir, 'config', 'panda.srdf')
        )
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Gazebo server only — gzclient skipped to avoid GPU/memory crashes
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': os.path.join(bringup_dir, 'worlds', 'sorting_world.sdf'),
            'verbose': 'false',
        }.items(),
    )

    # Spawn robot into Gazebo from robot_description topic
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'panda'],
        output='screen',
    )

    # Robot state publisher — full URDF with ros2_control tags intact
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': urdf_str}],
    )

    # MoveIt2 move_group action server
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict()],
    )

    # RViz with MoveIt2 config
    rviz_config = os.path.join(
        get_package_share_directory('moveit_resources_panda_moveit_config'),
        'launch', 'moveit.rviz'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
    )

    # Controller spawners — connect to Gazebo's controller manager
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )
    panda_arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['panda_arm_controller', '-c', '/controller_manager'],
    )
    panda_hand_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['panda_hand_controller', '-c', '/controller_manager'],
    )

    return LaunchDescription([
        gzserver,
        robot_state_publisher,
        spawn_robot,
        move_group_node,
        rviz_node,
        joint_state_broadcaster_spawner,
        panda_arm_controller_spawner,
        panda_hand_controller_spawner,
    ])