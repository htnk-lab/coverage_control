import os
from glob import glob

from setuptools import setup

package_name = "coverage_control"
coverage_utils = package_name + "/coverage_utils"


setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name, coverage_utils],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Toshiyuki Oshima",
    maintainer_email="toshiyuki67026@gmail.com",
    description="Package for experiencing coverage control",
    license="Apache License2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "agent_body = coverage_control.agent_body:main",
            "central = coverage_control.central:main",
            "controller = coverage_control.controller:main",
            "phi_marker_visualizer = coverage_control.phi_marker_visualizer:main",
            "phi_pointcloud_visualizer = coverage_control.phi_pointcloud_visualizer:main",
            "pose_collector = coverage_control.pose_collector:main",
            "vel_collector = coverage_control.vel_collector:main",
            "sensing_region_marker_visualizer = coverage_control.sensing_region_marker_visualizer:main",
            "sensing_region_pointcloud_visualizer = coverage_control.sensing_region_pointcloud_visualizer:main",
        ],
    },
)
