#!/usr/bin/env python

import traceback

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Float32MultiArray, Header
from visualization_msgs.msg import Marker

from .coverage_utils.field_generator import FieldGenerator
from .coverage_utils.utils import multiarray_to_ndarray, padding


class PhiMarkerVisualizer(Node):
    """Markerを用いて重要度分布を可視化

    Note:
        grid_mapは以下の並びとなるよう，reshapeと転置により整形
        self.rows = [
            [x1, y1, ...],
            [x2, y2, ...]
                :
            [xn, yn, ...]]
    """

    def __init__(self) -> None:
        super().__init__("phi_marker_visualizer")

        # declare parameter
        self.declare_parameter(
            "world_frame", "world", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
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

        # get parameter
        world_frame = str(self.get_parameter("world_frame").value)
        grid_accuracy = np.array(self.get_parameter("grid_accuracy").value)
        dim = len(grid_accuracy)
        limit = np.array(
            [
                self.get_parameter("x_limit").value,
                self.get_parameter("y_limit").value,
                self.get_parameter("z_limit").value,
            ]
        )

        # 次元に応じて適切な透過度を選択
        self.alpha = 0.7 - 0.2 * dim

        field_generator = FieldGenerator(grid_accuracy=grid_accuracy, limit=limit)
        grid_map = field_generator.generate_grid_map()

        self.rows: NDArray = np.array(grid_map).reshape([dim, -1]).T

        # 重要度分布描画用のMarkerを作成
        self.phi_marker = Marker(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=world_frame,
            ),
            ns="phi_marker",
            action=Marker.ADD,
            type=Marker.CUBE_LIST,
            scale=Vector3(
                **dict(zip(["x", "y", "z"], padding(original_array=field_generator.grid_span, padding_value=0.05)))
            ),
        )

        # pub
        self.phi_marker_pub = self.create_publisher(Marker, "/phi_marker", 10)

        # sub
        self.create_subscription(Float32MultiArray, "/phi", self.phi_callback, 10)

    def phi_callback(self, msg: Float32MultiArray) -> None:
        phi = multiarray_to_ndarray(float, np.float32, msg).reshape([-1, 1])

        # 次元を調整しつつmatplotlibのcolor_mapを利用して，重要度を色で表現
        rgba_phi: NDArray = plt.get_cmap("jet")(phi).squeeze()

        self.phi_marker.header.stamp = self.get_clock().now().to_msg()
        # 2次元以下の場合は足りない座標分を0埋め
        self.phi_marker.points = [
            Point(**dict(zip(["x", "y", "z"], point))) for point in [padding(point) for point in self.rows]
        ]
        self.phi_marker.colors = [
            ColorRGBA(**dict(zip(["r", "g", "b", "a"], [*rgb, self.alpha]))) for rgb in rgba_phi[:, 0:3]
        ]
        self.phi_marker_pub.publish(self.phi_marker)


def main() -> None:
    rclpy.init()
    phi_marker_visualizer = PhiMarkerVisualizer()

    try:
        rclpy.spin(phi_marker_visualizer)
    except:
        phi_marker_visualizer.get_logger().error(traceback.format_exc())
    finally:
        phi_marker_visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
