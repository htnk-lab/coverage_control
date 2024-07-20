#!/usr/bin/env python3

from typing import List

from launch_ros.actions import Node, PushRosNamespace, SetParameter, SetParametersFromFile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration


def launch_setup(context: LaunchContext) -> List[GroupAction]:
    base_config = LaunchConfiguration("base_config")
    agent_config = LaunchConfiguration("agent_config")

    # LaunchConfigurationの中身を取得
    agent_prefix = LaunchConfiguration("agent_prefix", default="agent").perform(context)
    agent_id = LaunchConfiguration("agent_id").perform(context)
    agent_name = agent_prefix + agent_id

    # GroupActionによりnamespaceやparameterを一括して与える
    agent_group = GroupAction(
        actions=[
            PushRosNamespace(agent_name),
            SetParameter(name="agent_id", value=agent_id),
            SetParametersFromFile(base_config),
            SetParametersFromFile(agent_config),
            Node(
                package="cbf_pcc",
                executable="field_cbf_optimizer",
                output="screen",
            ),
            Node(
                package="cbf_pcc",
                executable="agent_body",
                remappings=[("cmd_vel" ,"cmd_vel_opt")],
            ),
            Node(
                package="cbf_pcc",
                executable="controller",
                remappings=[("cmd_vel" ,"cmd_vel_nom")],
            ),
            Node(
                package="cbf_pcc",
                executable="sensing_region_marker_visualizer",
            ),
            Node(
                package="cbf_pcc",
                executable="sensing_region_pointcloud_visualizer",
            ),
        ],
    )
    return [agent_group]


def generate_launch_description() -> LaunchDescription:
    DeclareLaunchArgument("base_config")
    DeclareLaunchArgument("agent_config")
    DeclareLaunchArgument("agent_prefix")
    DeclareLaunchArgument("agent_id")

    # Create the launch description and populate
    ld = LaunchDescription()

    # LaunchConfigurationの値を取得するため，OpaqueFunctionで外部実行
    # ref: https://answers.ros.org/question/340705/access-launch-argument-in-launchfile-ros2/
    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld
