#!/usr/bin/env python

import traceback

import matplotlib.colors as mcolors
import numpy as np
import rclpy
from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Int8MultiArray

from .coverage_utils.field_generator import FieldGenerator
from .coverage_utils.utils import color_list, multiarray_to_ndarray


class SensingRegionPointCloudVisualizer(Node):
    """PointCloud2を用いてセンシング領域を可視化"""

    def __init__(self) -> None:
        super().__init__("sensing_region_pointcloud_visualizer")

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

        # センシング領域描画用のpointcloudを作成
        self.sensing_region_pointcloud = PointCloud2(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=world_frame,
            ),
            height=1,
            fields=[
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="r", offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name="g", offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name="b", offset=20, datatype=PointField.FLOAT32, count=1),
            ],
            is_bigendian=False,
            point_step=24,
            is_dense=True,
        )

        # pub
        self.sensing_region_pointcloud_pub = self.create_publisher(PointCloud2, "sensing_region_pointcloud", 10)

        # sub
        self.create_subscription(Int8MultiArray, "sensing_region", self.sensing_region_callback, 10)

    def sensing_region_callback(self, msg: Int8MultiArray) -> None:
        """Int8Multiarrayからpointcloudを生成してpublish

        Args:
            msg (Int8MultiArray): センシング領域

        Note:
            センシング領域の格子点計算に際してはpointcloud生成用にreshapeと転置により整形

            points = [
            [x1, y1, z1, r1, g1, b1],
            [x2, y2, z2, r2, g2, b2],
                    :
            [xn, yn, zn, rn, gn, bn]]
        """
        sensing_region = multiarray_to_ndarray(bool, np.bool_, msg)
        sensing_region_grid_points = (
            np.array([self.grid_map[i][sensing_region] for i in range(self.dim)]).reshape(self.dim, -1).T
        )

        # np.repeatのために2次元配列として作成
        rgba_agent: NDArray = np.array([mcolors.to_rgba(color_list[self.agent_id])])
        assert rgba_agent.shape == (1, 4)

        # 2次元以下の場合は足りない座標分を0埋め
        points = np.hstack(
            [
                sensing_region_grid_points,
                np.zeros([sensing_region_grid_points.shape[0], 3 - self.dim]),
                np.repeat(rgba_agent[:, 0:3], sensing_region_grid_points.shape[0], axis=0),
            ]
        ).astype(np.float32)

        num_points = len(points)
        self.sensing_region_pointcloud.width = num_points
        self.sensing_region_pointcloud.row_step = self.sensing_region_pointcloud.point_step * num_points
        self.sensing_region_pointcloud.data = points.tobytes()
        self.sensing_region_pointcloud_pub.publish(self.sensing_region_pointcloud)


def main() -> None:
    rclpy.init()
    sensing_region_pointcloud_visualizer = SensingRegionPointCloudVisualizer()

    try:
        rclpy.spin(sensing_region_pointcloud_visualizer)
    except:
        sensing_region_pointcloud_visualizer.get_logger().error(traceback.format_exc())
    finally:
        sensing_region_pointcloud_visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
