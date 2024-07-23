import traceback
from typing import List, Tuple

import numpy as np
import rclpy
from cbfpy.cbf import CBFBase
from cbfpy.cbf_qp_solver import CBFNomQPSolver
from geometry_msgs.msg import Point, Pose, Twist, Vector3
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import Header
from sympy import Matrix, Symbol, lambdify, symbols
from visualization_msgs.msg import Marker

from .coverage_utils.utils import get_color_rgba

class WallCBF(CBFBase):
    def __init__(self) -> None:
        self.x = Matrix(symbols("x, y", real=True))
        self.sign = Symbol("sign_", real=True)
        # 初期化
        # self.segment_p1 = None
        # self.segment_p2 = None
        self.segment_p1 = Matrix(symbols("x1, y1", real=True))
        self.segment_p2 = Matrix(symbols("x2, y2", real=True))

    def _alpha(self, h: List[float]) -> List[float]:
        return [300*(np.exp(_h)-1) for _h in h]

    def set_parameters(self, d: float, keep_inside: bool = True) -> None:
        self.d = d
        self.keep_inside = keep_inside
        # self.segment_p1 = Matrix(symbols("x1, y1", real=True))
        # self.segment_p2 = Matrix(symbols("x2, y2", real=True))
        # sign = 1 if keep_inside else -1

        cbf_segment = self.sign * (self.distance_to_segment(self.x, self.segment_p1, self.segment_p2) - self.d**2)
        self._calc_dhdx_segment = lambdify([self.x, self.sign, self.segment_p1, self.segment_p2], cbf_segment.diff(self.x))
        self._calc_h_segment = lambdify([self.x, self.sign, self.segment_p1, self.segment_p2], cbf_segment)

        # cbf_end1 = self.sign * (self.distance_to_end(self.x, self.segment_p1, self.segment_p2)[0] - self.d**2)
        # self._calc_dhdx_end1 = lambdify([self.x, self.sign, self.segment_p1, self.segment_p2], cbf_end1.diff(self.x))
        # self._calc_h_end1 = lambdify([self.x, self.sign, self.segment_p1, self.segment_p2], cbf_end1)

    def get_parameters(self) -> Tuple[float, bool]:
        return self.d, self.keep_inside

    def calc_constraints(self, agent_position: np.ndarray, x_wall: np.ndarray, y_wall: np.ndarray) -> None:
        self.G = []
        self.h = []
        
        for i in range(len(x_wall)-1):
        # i = 1
            p1 = np.array([x_wall[i], y_wall[i]])
            p2 = np.array([x_wall[i + 1], y_wall[i + 1]])

            # エージェントの位置が線分の内側にあるか確認
            if self.inside_segment(agent_position, p1, p2):
                agent_position = agent_position.flatten()
                sign = 1 if self.keep_inside else -1

                G_segment = self._calc_dhdx_segment(agent_position, sign, p1, p2).flatten()
                h_segment = self._calc_h_segment(agent_position, sign, p1, p2)

                self.G.append(G_segment)
                self.h.append(h_segment)

            # else:
            #     agent_position = agent_position.flatten()
            #     sign = 1 if self.keep_inside else -1

            #     G_end = self._calc_dhdx_end(agent_position, sign, p1, p2).flatten()
            #     h_end = self._calc_h_end(agent_position, sign, p1, p2)

            #     self.G.append(G_end)
            #     self.h.append(h_end)

        
        sort_h = sorted(self.h)
        sort_index_h = sorted(range(len(self.h)), key=lambda k: self.h[k])

        # 制約を持つ線分の数を選択
        number_constrainted_segment = 2

        self.h = sort_h[0:number_constrainted_segment]
        sort_index = sort_index_h[0:number_constrainted_segment]
        self.G = [self.G[i] for i in sort_index]

        print(self.G)
        print(self.h)

        if len(self.G) == 0:
            self.G = [np.zeros((2,))]
            self.h = [np.ones((1,))]

    def distance_to_segment(self, point: Matrix, p1: np.ndarray, p2: np.ndarray) -> Symbol:
        x1, y1 = p1
        x2, y2 = p2
        px, py = point
        dx = x2 - x1
        dy = y2 - y1
        a = -dy
        b = dx
        c = -x2*y1 + y2*x1
        d = dx**2 + dy**2
        distance = (a*px + b*py + c)**2 / d
        return distance
    
    def distance_to_end(self, point: Matrix, p1: np.ndarray, p2: np.ndarray) -> Symbol:
        x1, y1 = p1
        x2, y2 = p2
        px, py = point        
        distance_1 = (px - x1)**2 + (py - y1)**2
        distance_2 = (px - x2)**2 + (py - y2)**2
        return distance_1, distance_2
    
    def inside_segment(self, agent_position: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> bool:
        inner_product = (agent_position[0] - p1[0]) * (p2[0] - p1[0]) + (agent_position[1] - p1[1]) * (p2[1] - p1[1])
        if inner_product < 0:
            return False  # 線分の外側にある

        length = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
        if inner_product >= length:
            return False  # 線分の外側にある
        return True  # 線分の内側にある



class CBFOptimizer:
    def __init__(self) -> None:
        self.qp_nom_solver = CBFNomQPSolver()
        self.P = np.eye(2)

        self.wall_cbf = WallCBF()

        # initialize(must be overwritten)
        self.set_parameters(0.01, True)

    def set_parameters(self, d: float, keep_inside: bool = True) -> None:
        self.wall_cbf.set_parameters(d, keep_inside)

    def get_parameters(self) -> Tuple[float, bool]:
        return self.wall_cbf.get_parameters()

    def _calc_constraints(self, agent_position: np.ndarray, x_wall: np.ndarray, y_wall: np.ndarray) -> None:
        self.wall_cbf.calc_constraints(agent_position, x_wall, y_wall)

    def _get_constraints(self) -> Tuple[List[np.ndarray], List[float]]:
        G, alpha_h = self.wall_cbf.get_constraints()
        return G, alpha_h

    def optimize(
        self, nominal_input: np.ndarray, agent_position: np.ndarray, x_wall: np.ndarray, y_wall: np.ndarray
    ) -> Tuple[str, np.ndarray]:
        self._calc_constraints(agent_position, x_wall, y_wall)
        G_list, alpha_h_list = self._get_constraints()
        # if not alpha_h_list == []
        self.qp_nom_solver._set_solver_options(feastol=1e-14)
        # print(G_list)
        # print(alpha_h_list)

        try:
            return self.qp_nom_solver.optimize(nominal_input, self.P, G_list, alpha_h_list)
        except Exception as e:
            return 0, nominal_input


class FieldCBFOptimizer(Node):
    def __init__(self) -> None:
        super().__init__("field_cbf_optimizer")

        # declare parameter
        self.declare_parameter(
            "world_frame", "world", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter("activate_cbf", True, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("x_wall", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter("y_wall", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter(
            "cbf_keep_inside", True, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)
        )

        # get parameter
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.activate_cbf = bool(self.get_parameter("activate_cbf").value)
        self.x_wall = np.array(self.get_parameter("x_wall").value)
        self.y_wall = np.array(self.get_parameter("y_wall").value)
        self.keep_inside = bool(self.get_parameter("cbf_keep_inside").value)

        self.optimizer = CBFOptimizer()

        self.curr_pose = Pose()

        # 障害物を表示
        self.obstacle_marker = Marker(
            header=Header(stamp=self.get_clock().now().to_msg(), frame_id=self.world_frame),
            ns="obstacle_marker",
            action=Marker.ADD,
            type=Marker.LINE_LIST,
            scale=Vector3(x=0.03, y=0.0, z=0.0),
            color=get_color_rgba(color="r", alpha=0.7),
            points=[],
        )
        self._update_obstacle_marker()

        # pub
        self.cmd_vel_opt_pub = self.create_publisher(Twist, "cmd_vel_opt", 10)
        self.obstacle_marker_pub = self.create_publisher(Marker, "obstacle_marker", 10)

        # sub
        self.create_subscription(Pose, "curr_pose", self.curr_pose_callback, 10)
        self.create_subscription(Twist, "cmd_vel_nom", self.cmd_vel_nom_callback, 10)

    def _update_obstacle_marker(self) -> None:
        points = []
        for i in range(len(self.x_wall) - 1):
            points.append(Point(x=self.x_wall[i], y=self.y_wall[i], z=0.0))
            points.append(Point(x=self.x_wall[i + 1], y=self.y_wall[i + 1], z=0.0))
        self.obstacle_marker.points = points

    def curr_pose_callback(self, msg: Pose) -> None:
        self.curr_pose = msg
        # self.get_logger().info(f"{self.curr_pose}")

    def cmd_vel_nom_callback(self, msg: Twist) -> None:
        cmd_vel_opt = msg

        # optimize
        if self.activate_cbf:
            self.optimizer.set_parameters(d=0.001, keep_inside=self.keep_inside)
            agent_position = np.array([self.curr_pose.position.x, self.curr_pose.position.y])
            nominal_input = np.array([msg.linear.x, msg.linear.y])
            # nominal_input = np.array([0.0, 0.0])

            # 最適入力を計算
            _, optimal_input = self.optimizer.optimize(nominal_input, agent_position, self.x_wall, self.y_wall)
            # self.get_logger().info(f"{self.optimizer.wall_cbf.h}")
            # self.get_logger().info(f"{self.optimizer.wall_cbf.get_constraints()}")
            # G,alpha_h=self.optimizer.wall_cbf.get_constraints()
            # if isinstance(G,list):
            #     G=np.array(G)
            #     alpha_h=np.array(alpha_h)
                # self.get_logger().info(f":{G@optimal_input}")
                # self.get_logger().info(f"h:{-alpha_h}")


            # self.optimizer.optimize(nominal_input, agent_position, self.x_wall, self.y_wall)
            # cmd_vel_opt.linear = Vector3(x=float(nominal_input[0]), y=float(nominal_input[1]))
            cmd_vel_opt.linear = Vector3(x=float(optimal_input[0]), y=float(optimal_input[1]))
            # self.get_logger().info(f"{cmd_vel_opt.linear}")

        self.cmd_vel_opt_pub.publish(cmd_vel_opt)

        self.obstacle_marker.header.stamp = self.get_clock().now().to_msg()
        self.obstacle_marker_pub.publish(self.obstacle_marker)


def main() -> None:
    rclpy.init()
    field_cbf_optimizer = FieldCBFOptimizer()

    try:
        rclpy.spin(field_cbf_optimizer)
    except:
        field_cbf_optimizer.get_logger().error(traceback.format_exc())
    finally:
        field_cbf_optimizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
