#!/usr/bin/env python

import traceback
from dataclasses import dataclass
from functools import partial
from typing import List

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import Header


@dataclass
class Data:
    curr_pose: Pose = Pose()
    is_ready: bool = False


class PoseCollector(Node):
    """エージェントの位置姿勢を収集"""

    def __init__(self) -> None:
        super().__init__("pose_collector")

        # declare parameter
        self.declare_parameter("world_frame", "world", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("agent_num", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("agent_prefix", "agent", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("timer_period", 0.1, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        self.world_frame = str(self.get_parameter("world_frame").value)
        agent_num = int(self.get_parameter("agent_num").value)
        agent_prefix = str(self.get_parameter("agent_prefix").value)

        self.data_list: List[Data] = [Data()] * agent_num
        self.is_ready = False

        timer_period = float(self.get_parameter("timer_period").value)

        # pub
        self.curr_pose_array_pub = self.create_publisher(PoseArray, "curr_pose_array", 10)

        # sub
        # agentの数とnamespaceに対応してsubscriptionとcallbackを登録
        for agent_id in range(agent_num):
            agent_name = agent_prefix + str(agent_id)
            topic_name = agent_name + "/curr_pose"
            self.create_subscription(Pose, topic_name, partial(self.curr_pose_callback, agent_id=agent_id), 10)

        # timer
        self.create_timer(timer_period, self.timer_callback)

    def curr_pose_callback(self, msg: Pose, agent_id: int) -> None:
        assert isinstance(agent_id, int)
        self.data_list[agent_id] = Data(curr_pose=msg, is_ready=True)

    def timer_callback(self) -> None:
        # 全agentのcurr_poseが揃い，is_ready==Trueとなるまでpublishしない
        if self.is_ready:
            # agent_id順に現在位置を格納
            curr_pose_array = [data.curr_pose for data in self.data_list]
            self.curr_pose_array_pub.publish(
                PoseArray(
                    header=Header(stamp=self.get_clock().now().to_msg(), frame_id=self.world_frame),
                    poses=curr_pose_array,
                )
            )
        else:
            if len([data.is_ready for data in self.data_list if not data.is_ready]) == 0:
                self.get_logger().warn("pose_collector is ready")
                self.is_ready = True


def main() -> None:
    rclpy.init()
    pose_collector = PoseCollector()

    try:
        rclpy.spin(pose_collector)
    except:
        pose_collector.get_logger().error(traceback.format_exc())
    finally:
        pose_collector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
