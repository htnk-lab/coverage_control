#!/usr/bin/env python

import traceback
from dataclasses import dataclass
from functools import partial
from typing import List

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


@dataclass
class Data:
    curr_vel: Float32MultiArray = Float32MultiArray()
    is_ready: bool = False


class VelCollector(Node):
    """エージェントの速度を収集"""

    def __init__(self) -> None:
        super().__init__("vel_collector")

        # declare parameter
        self.declare_parameter(
            "world_frame", "world", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter("agent_num", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter(
            "agent_prefix", "agent", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            "timer_period", 0.1, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        )

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.agent_num = int(self.get_parameter("agent_num").value)
        agent_prefix = str(self.get_parameter("agent_prefix").value)

        self.data_list: List[Data] = [Data()] * self.agent_num
        self.is_ready = False

        timer_period = float(self.get_parameter("timer_period").value)

        # pub
        self.curr_vel_array_pub = self.create_publisher(Float32MultiArray, "curr_vel_array", 10)

        # sub
        # agentの数とnamespaceに対応してsubscriptionとcallbackを登録
        for agent_id in range(self.agent_num):
            agent_name = agent_prefix + str(agent_id)
            topic_name = agent_name + "/curr_vel"
            self.create_subscription(Twist, topic_name, partial(self.curr_vel_callback, agent_id=agent_id), 10)

        # timer
        self.create_timer(timer_period, self.timer_callback)

    def curr_vel_callback(self, msg: Twist, agent_id: int) -> None:
        assert isinstance(agent_id, int)
        curr_velocity = Float32MultiArray(data=[msg.linear.x, msg.linear.y])
        self.data_list[agent_id] = Data(curr_vel=curr_velocity, is_ready=True)

    def timer_callback(self) -> None:
        # 全agentのcurr_velが揃い，is_ready==Trueとなるまでpublishしない
        if self.is_ready:
            curr_vel_array = []

            for data in self.data_list:
                curr_vel_array.extend(data.curr_vel.data)
            dim = MultiArrayDimension()
            dim.size = len(curr_vel_array)  # 要素数を設定
            dim.label = ""  # 任意のラベルを設定
            curr_vel_multiarray = Float32MultiArray(data=curr_vel_array)
            curr_vel_multiarray.layout.dim = [dim]
            self.curr_vel_array_pub.publish(curr_vel_multiarray)
        else:
            if len([data.is_ready for data in self.data_list if not data.is_ready]) == 0:
                self.get_logger().warn("vel_collector is ready")
                self.is_ready = True


def main() -> None:
    rclpy.init()
    vel_collector = VelCollector()

    try:
        rclpy.spin(vel_collector)
    except:
        vel_collector.get_logger().error(traceback.format_exc())
    finally:
        vel_collector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
