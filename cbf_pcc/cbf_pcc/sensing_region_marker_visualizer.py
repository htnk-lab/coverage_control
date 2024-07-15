#!/usr/bin/env python

import traceback

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import Header, Int8MultiArray
from visualization_msgs.msg import Marker

from .coverage_utils.field_generator import FieldGenerator
from .coverage_utils.utils import color_list, get_color_rgba, multiarray_to_ndarray, padding


class SensingRegionMarkerVisualizer(Node):
    """Markerを用いてセンシング領域を可視化"""

    def __init__(self) -> None:
        super().__init__("sensing_region_marker_visualizer")

        # declare parameter
        self.declare_parameter(
            "world_frame", "world", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter("agent_id", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
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
        self.agent_id = int(self.get_parameter("agent_id").value)

        grid_accuracy = np.array(self.get_parameter("grid_accuracy").value)
        self.dim = len(grid_accuracy)
        limit = np.array(
            [
                self.get_parameter("x_limit").value,
                self.get_parameter("y_limit").value,
                self.get_parameter("z_limit").value,
            ]
        )

        field_generator = FieldGenerator(grid_accuracy=grid_accuracy, limit=limit)
        self.grid_map = field_generator.generate_grid_map()
        # 次元に応じて適切な透過度を選択
        self.alpha = 0.7 - 0.15 * self.dim

        # センシング領域描画用のMarkerを作成
        self.sensing_region_marker = Marker(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=world_frame,
            ),
            ns="sensing_region_marker",
            action=Marker.ADD,
            type=Marker.CUBE_LIST,
            scale=Vector3(
                **dict(
                    zip(
                        ["x", "y", "z"],
                        padding(
                            original_array=field_generator.grid_span,
                            padding_value=0.05,
                        ),
                    )
                )
            ),
        )

        # pub
        self.sensing_region_marker_pub = self.create_publisher(Marker, "sensing_region_marker", 10)

        # sub
        self.create_subscription(Int8MultiArray, "sensing_region", self.sensing_region_callback, 10)

    def sensing_region_callback(self, msg: Int8MultiArray) -> None:
        """Int8Multiarrayからpointcloudを生成してpublish

        Args:
            msg (Int8MultiArray): センシング領域

        Note:
            センシング領域の格子点計算に際しては，以下の並びになるようreshapeと転置により整形
            [[x1, y1, ...],
            [x2, y2, ...]
                :
            [xn, yn, ...]]
        """
        sensing_region = multiarray_to_ndarray(bool, np.bool_, msg)
        sensing_region_grid_points = (
            np.array([self.grid_map[i][sensing_region] for i in range(self.dim)]).reshape(self.dim, -1).T
        )

        self.sensing_region_marker.header.stamp = self.get_clock().now().to_msg()
        # 2次元以下の場合は足りない座標分を0埋め
        self.sensing_region_marker.points = [
            Point(**dict(zip(["x", "y", "z"], point)))
            for point in [padding(point) for point in sensing_region_grid_points]
        ]
        self.sensing_region_marker.colors = [
            get_color_rgba(color_list[self.agent_id], self.alpha)
        ] * sensing_region_grid_points.shape[0]

        self.sensing_region_marker_pub.publish(self.sensing_region_marker)


def main() -> None:
    rclpy.init()
    sensing_region_marker_visualizer = SensingRegionMarkerVisualizer()

    try:
        rclpy.spin(sensing_region_marker_visualizer)
    except:
        sensing_region_marker_visualizer.get_logger().error(traceback.format_exc())
    finally:
        sensing_region_marker_visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
