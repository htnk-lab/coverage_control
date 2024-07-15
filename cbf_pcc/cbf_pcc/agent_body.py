#!/usr/bin/env python

import traceback

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, TransformStamped, Twist, Vector3
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import Header
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class AgentBody(Node):
    """1次積分系の仮想エージェント"""

    def __init__(self) -> None:
        super().__init__("agent_body")

        # declare parameter
        self.declare_parameter(
            "init_position",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY),
        )
        self.declare_parameter("init_yaw", 0.0, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(
            "world_frame", "world", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            "agent_frame", "base", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter("dt", 0.1, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        # get parameter
        # curr_pose初期化
        # parameterのinit_positionの要素数が足りない場合，対応するPointの要素は0.0で初期化される
        self.curr_pose = Pose(
            position=Point(**dict(zip(["x", "y", "z"], self.get_parameter("init_position").value))),
            orientation=Quaternion(
                **dict(
                    zip(
                        ["x", "y", "z", "w"],
                        quaternion_from_euler(ai=0.0, aj=0.0, ak=float(self.get_parameter("init_yaw").value)),
                    )
                )
            ),
        )
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.agent_frame = str(self.get_namespace() + "/" + self.get_parameter("agent_frame").value)
        self.dt = float(self.get_parameter("dt").value)

        # tf2
        self.broadcaster = TransformBroadcaster(self)

        # pub
        self.curr_pose_pub = self.create_publisher(Pose, "curr_pose", 10)

        # sub
        self.create_subscription(Twist, "cmd_vel", self.cmd_vel_callback, 10)

    def cmd_vel_callback(self, msg: Twist) -> None:
        # 速度指令から積分系に基づき位置を更新
        position = self.curr_pose.position
        self.curr_pose.position = Point(
            x=position.x + self.dt * msg.linear.x,
            y=position.y + self.dt * msg.linear.y,
            z=position.z + self.dt * msg.linear.z,
        )

        # とりあえず姿勢はyawのみ更新
        orientation = self.curr_pose.orientation
        _, _, yaw = euler_from_quaternion(quaternion=[orientation.x, orientation.y, orientation.z, orientation.w])
        self.curr_pose.orientation = Quaternion(
            **dict(
                zip(
                    ["x", "y", "z", "w"],
                    quaternion_from_euler(ai=0.0, aj=0.0, ak=yaw + self.dt * msg.angular.z),
                )
            )
        )
        self.curr_pose_pub.publish(self.curr_pose)

        # 描画用に現在位置をtf形式で送信
        curr_pose = self.curr_pose
        transform_stamped = TransformStamped(
            header=Header(stamp=self.get_clock().now().to_msg(), frame_id=self.world_frame),
            child_frame_id=self.agent_frame,
            transform=Transform(
                translation=Vector3(
                    x=curr_pose.position.x,
                    y=curr_pose.position.y,
                    z=curr_pose.position.z,
                ),
                rotation=curr_pose.orientation,
            ),
        )
        self.broadcaster.sendTransform(transform_stamped)


def main() -> None:
    rclpy.init()
    agent_body = AgentBody()

    try:
        rclpy.spin(agent_body)
    except:
        agent_body.get_logger().error(traceback.format_exc())
    finally:
        agent_body.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
