#!/usr/bin/env python

import traceback
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, Twist, Vector3
from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int8MultiArray

from .coverage_utils.field_generator import FieldGenerator
from .coverage_utils.utils import multiarray_to_ndarray, ndarray_to_multiarray, padding
from .coverage_utils.voronoi import Voronoi


class Controller(Node):
    """エージェントのローカル制御器"""

    def __init__(self) -> None:
        super().__init__("controller")

        # declare parameter
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
        self.declare_parameter(
            "sensing_radius", 0.5, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        )
        self.declare_parameter("kp", 1.0, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(
            "timer_period", 0.1, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        )

        # get parameter
        self.agent_id = int(self.get_parameter("agent_id").value)

        # fieldを規定するパラメータを取得
        grid_accuracy = np.array(self.get_parameter("grid_accuracy").value)
        limit = np.array(
            [
                self.get_parameter("x_limit").value,
                self.get_parameter("y_limit").value,
                self.get_parameter("z_limit").value,
            ]
        )

        field_generator = FieldGenerator(grid_accuracy=grid_accuracy, limit=limit)

        # 重要度分布初期化
        self.phi = field_generator.generate_phi()
        self.grid_map = field_generator.generate_grid_map()
        self.point_density = np.prod(field_generator.grid_span)

        sensing_radius = float(self.get_parameter("sensing_radius").value)
        self.voronoi = Voronoi(radius=sensing_radius)  # r-limited voronoi

        self.ref_pose = Pose()
        self.curr_pose = Pose()
        self.kp = float(self.get_parameter("kp").value)
        timer_period = float(self.get_parameter("timer_period").value)

        # pub
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.sensing_region_pub = self.create_publisher(Int8MultiArray, "sensing_region", 10)

        # sub
        self.create_subscription(Pose, "curr_pose", self.curr_pose_callback, 10)
        self.create_subscription(PoseArray, "/curr_pose_array", self.curr_pose_array_callback, 10)
        self.create_subscription(Float32MultiArray, "/phi", self.phi_callback, 10)

        # timer
        self.create_timer(timer_period, self.timer_callback)

    def curr_pose_callback(self, msg: Pose) -> None:
        self.curr_pose = msg

    def curr_pose_array_callback(self, msg: PoseArray) -> None:
        """近隣エージェントの位置姿勢から自身のセンシング領域(ボロノイ領域)とその重心を計算して目標座標へ反映"""

        # 近隣エージェントの位置を用いてセンシング領域を計算
        centroid_position, sensing_region_grid_points, sensing_region = self.calc_voronoi_tesselation(msg.poses)

        # 各次元に対応して使わない要素を0で埋めた上で，unpackしたもの目標座標とする
        self.ref_pose = Pose(position=Point(**dict(zip(["x", "y", "z"], padding(centroid_position)))))

        # センシング領域をpublish
        self.sensing_region_pub.publish(ndarray_to_multiarray(Int8MultiArray, sensing_region))

    def calc_voronoi_tesselation(self, pose_list: List[Pose]) -> Tuple[NDArray, List[NDArray], NDArray]:
        all_agent_position_list: List[NDArray] = []

        # Pose型のリストから各エージェント位置を抜き出してNDArray形式のリストへ変換，格納
        for pose in pose_list:
            all_agent_position_list.append(np.array([pose.position.x, pose.position.y, pose.position.z]))

        agent_position = all_agent_position_list[self.agent_id]

        # 自分の位置のみ除外
        neighbor_agent_position_list = [
            neighbor_agent_position
            for agent_id, neighbor_agent_position in enumerate(all_agent_position_list)
            if agent_id != self.agent_id
        ]

        # センシング領域を計算
        # 円形センサーモデル使用時は，性能関数としてK.Sugimoto et al. 2015のh_{1}を採用した場合の最適解に相当
        centroid_position, sensing_region_grid_points, sensing_region = self.voronoi.calc_tesselation(
            agent_position=agent_position,
            neighbor_agent_position_list=neighbor_agent_position_list,
            phi=self.phi,
            grid_map=self.grid_map,
            point_density=self.point_density,
        )
        return centroid_position, sensing_region_grid_points, sensing_region

    def phi_callback(self, msg: Float32MultiArray) -> None:
        self.phi = multiarray_to_ndarray(float, np.float32, msg)

    def timer_callback(self) -> None:
        # 位置偏差
        x_err = self.ref_pose.position.x - self.curr_pose.position.x
        y_err = self.ref_pose.position.y - self.curr_pose.position.y
        z_err = self.ref_pose.position.z - self.curr_pose.position.z

        # P制御
        cmd_vel = Twist(
            linear=Vector3(
                x=self.kp * x_err,
                y=self.kp * y_err,
                z=self.kp * z_err,
            )
        )
        self.cmd_vel_pub.publish(cmd_vel)


def main() -> None:
    rclpy.init()
    controller = Controller()

    try:
        rclpy.spin(controller)
    except:
        controller.get_logger().error(traceback.format_exc())
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
