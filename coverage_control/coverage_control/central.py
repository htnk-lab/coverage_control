#!/usr/bin/env python

import traceback
from enum import IntEnum

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .coverage_utils.field_generator import FieldGenerator
from .coverage_utils.utils import ndarray_to_multiarray, smooth_ramp


class InitPhiType(IntEnum):
    """重要度分布の初期化方法を定義"""

    UNIFORM = 1
    GAUSSIAN = 2
    DISK = 3


class Central(Node):
    """重要度分布を管理する集中制御器"""

    def __init__(self) -> None:
        super().__init__("central")

        # declare parameter
        self.declare_parameter(
            "grid_accuracy", [10, 10], descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY)
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

        timer_period = float(self.get_parameter("timer_period").value)

        # pub
        self.phi_pub = self.create_publisher(Float32MultiArray, "/phi", 10)

        # timer
        self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self) -> None:
        """生成した重要度分布をpublish"""
        self.phi_pub.publish(ndarray_to_multiarray(Float32MultiArray, self.phi))


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
