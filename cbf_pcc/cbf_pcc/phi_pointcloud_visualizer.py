#!/usr/bin/env python

import traceback

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, Header

from .coverage_utils.field_generator import FieldGenerator
from .coverage_utils.utils import multiarray_to_ndarray


class PhiPointCloudVisualizer(Node):
    """PointCloud2を用いて重要度分布を可視化

    Note:
        grid_mapは以下の並びとなるよう，reshapeと転置により整形
        self.rows = [
            [x1, y1, ...],
            [x2, y2, ...]
                :
            [xn, yn, ...]]
    """

    def __init__(self) -> None:
        super().__init__("phi_pointcloud_visualizer")

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
        self.dim = len(self.get_parameter("grid_accuracy").value)
        limit = np.array(
            [
                self.get_parameter("x_limit").value,
                self.get_parameter("y_limit").value,
                self.get_parameter("z_limit").value,
            ]
        )

        field_generator = FieldGenerator(grid_accuracy=grid_accuracy, limit=limit)
        grid_map = field_generator.generate_grid_map()
        self.rows: NDArray = np.array(grid_map).reshape([self.dim, -1]).T

        # 重要度分布描画用のpointcloudを作成
        self.phi_pointcloud = PointCloud2(
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
        self.phi_pointcloud_pub = self.create_publisher(PointCloud2, "phi_pointcloud", 10)

        # sub
        self.create_subscription(Float32MultiArray, "phi", self.phi_callback, 10)

    def phi_callback(self, msg: Float32MultiArray) -> None:
        """Float32Multiarrayからpointcloudを生成してpublish

        Args:
            msg (Float32MultiArray): 重要度分布

        Note:
            points = [
            [x1, y1, z1, r1, g1, b1],
            [x2, y2, z2, r2, g2, b2],
                    :
            [xn, yn, zn, rn, gn, bn]]

            pointsのz座標等にphiを反映することで，座標でも重要度を表現可能
        """
        phi = multiarray_to_ndarray(float, np.float32, msg).reshape([-1, 1])

        # 次元を調整しつつmatplotlibのcolor_mapを利用して，重要度を色で表現
        rgba_phi: NDArray = plt.get_cmap("jet")(phi).squeeze()

        # 2次元以下の場合は足りない座標分を0埋め
        points = np.hstack(
            [
                self.rows,
                np.zeros([self.rows.shape[0], 3 - self.dim]),
                rgba_phi[:, 0:3],
            ]
        ).astype(np.float32)

        num_points = len(points)
        self.phi_pointcloud.width = num_points
        self.phi_pointcloud.row_step = self.phi_pointcloud.point_step * num_points
        self.phi_pointcloud.data = points.tobytes()
        self.phi_pointcloud_pub.publish(self.phi_pointcloud)


def main() -> None:
    rclpy.init()
    phi_pointcloud_visualizer = PhiPointCloudVisualizer()

    try:
        rclpy.spin(phi_pointcloud_visualizer)
    except:
        phi_pointcloud_visualizer.get_logger().error(traceback.format_exc())
    finally:
        phi_pointcloud_visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
