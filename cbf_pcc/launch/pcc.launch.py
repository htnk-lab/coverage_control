#!/usr/bin/env python3

import os
from typing import List, Union

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, SetParametersFromFile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context: LaunchContext) -> List[Union[Node, GroupAction, LaunchDescription]]:
    dim = int(LaunchConfiguration("dim", default=2).perform(context))
    assert dim in range(1, 4), f"invalid dimension: {dim}"
    agent_num = int(LaunchConfiguration("num", default=3).perform(context))
    assert agent_num in range(1, 6), f"invalid agent_num: {agent_num}"
    init_phi_type = int(LaunchConfiguration("phi", default=1).perform(context))
    assert init_phi_type in range(1, 4), f"invalid init_phi_type: {init_phi_type}"

    pkg_cbf_pcc = get_package_share_directory("cbf_pcc")
    rviz_config = os.path.join(pkg_cbf_pcc, "rviz", f"{dim}d.rviz")
    assert os.path.exists(rviz_config)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
    )

    base_config = os.path.join(pkg_cbf_pcc, "config", "base.params.yaml")
    agent_config = os.path.join(pkg_cbf_pcc, "config", f"{dim}d.params.yaml")

    # group actionでまとめることでconfigを共通で与える
    central_group = GroupAction(
        actions=[
            SetParametersFromFile(agent_config),
            SetParametersFromFile(base_config),
            Node(
                package="cbf_pcc",
                executable="pose_collector",
                parameters=[{"agent_num": agent_num}],
                output="screen",
            ),
            Node(
                package="cbf_pcc",
                executable="central",
                parameters=[{"init_phi_type": init_phi_type, "agent_num": agent_num}],
            ),
            Node(package="cbf_pcc", executable="phi_marker_visualizer"),
            Node(package="cbf_pcc", executable="phi_pointcloud_visualizer"),
        ]
    )

    agent_launch_list = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(pkg_cbf_pcc, "launch", "agent.launch.py")]),
            launch_arguments={
                "base_config": base_config,
                "agent_config": agent_config,
                "agent_id": str(agent_id),
            }.items(),
        )
        for agent_id in range(agent_num)
    ]
    # nodeの起動順に起因するagentのジャンプを防ぐため，central系を後に
    return agent_launch_list + [rviz_node, central_group]


def generate_launch_description() -> LaunchDescription:
    DeclareLaunchArgument("dim", description="\in {1, 2, 3}")
    DeclareLaunchArgument("num", description="< 6")
    DeclareLaunchArgument("phi", description="\in {1, 2, 3}")

    # Create the launch description and populate
    ld = LaunchDescription()

    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld
