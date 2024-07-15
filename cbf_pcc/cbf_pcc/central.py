#!/usr/bin/env python

import traceback
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import List

import numpy as np
import rclpy
from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int8MultiArray

from .coverage_utils.field_generator import FieldGenerator
from .coverage_utils.utils import multiarray_to_ndarray, ndarray_to_multiarray, smooth_ramp


class InitPhiType(IntEnum):
    """重要度分布の初期化方法を定義"""

    UNIFORM = 1
    GAUSSIAN = 2
    DISK = 3


@dataclass
class Data:
    """センシング領域管理用"""

    sensing_region: NDArray
    is_ready: bool = False


class Central(Node):
    """重要度分布を管理する集中制御器"""

    def __init__(self) -> None:
        super().__init__("central")

        # declare parameter
        self.declare_parameter(
            "grid_accuracy", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY)
        )
        self.declare_parameter(
            "x_limit", [-1.0, 1.0], descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)
        )
        self.declare_parameter(
            "y_limit", [-1.0, 1.0], descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)
        )
        self.declare_parameter(
            "z_limit", [-1.0, 1.0], descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)
        )
        self.declare_parameter(
            "init_phi_type", 1, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER)
        )
        self.declare_parameter(
            "delta_decrease", 0.0, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        )
        self.declare_parameter(
            "delta_increase", 0.0, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        )
        self.declare_parameter("agent_num", -1, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter(
            "agent_prefix", "agent_", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            "timer_period", 0.1, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        )

        # get parameter
        grid_accuracy = np.array(self.get_parameter("grid_accuracy").value)
        self.dim = len(self.get_parameter("grid_accuracy").value)
        limit = np.array(
            [
                self.get_parameter("x_limit").value,
                self.get_parameter("y_limit").value,
                self.get_parameter("z_limit").value,
            ]
        )
        center = np.sum(limit, axis=1) / 2.0
        width = np.diff(limit, axis=1)
        init_phi_type = InitPhiType(int(self.get_parameter("init_phi_type").value))
        self.get_logger().info(f"init_phi_type: {init_phi_type.name}")

        field_generator = FieldGenerator(grid_accuracy=grid_accuracy, limit=limit)
        self.phi = field_generator.generate_phi()
        self.grid_map = field_generator.generate_grid_map()

        # 重要度分布初期化
        if init_phi_type == InitPhiType.UNIFORM:
            self.phi = field_generator.generate_phi()
        elif init_phi_type == InitPhiType.GAUSSIAN:
            self.phi = np.exp(-((sum([(self.grid_map[i] - center[i]) ** 2 for i in range(self.dim)])) ** 2))
        elif init_phi_type == InitPhiType.DISK:
            self.phi = np.multiply.reduce(
                [
                    np.exp(-smooth_ramp((self.grid_map[i] - center[i]) ** 2 - (width[i] / 2.0 - 0.5) ** 2))
                    for i in range(self.dim)
                ]
            )
        self.delta_decrease = float(self.get_parameter("delta_decrease").value)
        self.delta_increase = float(self.get_parameter("delta_increase").value)

        self.agent_num = int(self.get_parameter("agent_num").value)
        agent_prefix = str(self.get_parameter("agent_prefix").value)

        self.is_ready = False
        self.data_list: List[Data] = [
            Data(sensing_region=np.zeros_like(self.grid_map[0], np.bool_)),
        ] * self.agent_num

        timer_period = float(self.get_parameter("timer_period").value)

        # pub
        self.phi_pub = self.create_publisher(Float32MultiArray, "/phi", 10)

        # sub
        # 理論上センシング領域はエージェントの位置と半径のみで計算されるが，
        # 実装上は各自のセンシング領域をsubscribeして合算する．
        for agent_id in range(self.agent_num):
            agent_name = agent_prefix + str(agent_id)
            topic_name = agent_name + "/sensing_region"
            self.create_subscription(
                Int8MultiArray, topic_name, partial(self.sensing_region_callback, agent_id=agent_id), 10
            )

        # timer
        self.create_timer(timer_period, self.timer_callback)

    def sensing_region_callback(self, msg: Int8MultiArray, agent_id: int) -> None:
        self.data_list[agent_id] = Data(
            sensing_region=multiarray_to_ndarray(bool, np.bool_, msg),
            is_ready=True,
        )

    def timer_callback(self) -> None:
        """重要度分布をpublish"""

        # 全エージェントにおけるセンシング領域がsubscribeされるまで重要度分布の更新を行わない
        if self.is_ready:
            sensing_region = np.zeros_like(self.grid_map[0], dtype=np.bool_)
            for data in self.data_list:
                sensing_region += data.sensing_region
                # 重要度分布を更新
            self.phi = self.update_phi(self.phi, sensing_region)
        else:
            if len([data.is_ready for data in self.data_list if not data.is_ready]) == 0:
                self.get_logger().warn("central is ready")
                self.is_ready = True

        self.phi_pub.publish(ndarray_to_multiarray(Float32MultiArray, self.phi))

    def update_phi(self, phi: NDArray, region: NDArray) -> NDArray:
        phi += -self.delta_decrease * phi * region + self.delta_increase * (1 - phi) * ~region
        phi = np.where(phi < 0.01, 0.01, phi)
        phi = np.where(phi > 1.0, 1.0, phi)
        return phi


def main() -> None:
    rclpy.init()
    central = Central()

    try:
        rclpy.spin(central)
    except:
        central.get_logger().error(traceback.format_exc())
    finally:
        central.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
