import os
from glob import glob

from setuptools import setup

package_name = "cbf_pcc"
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
            "agent_body = cbf_pcc.agent_body:main",
            "central = cbf_pcc.central:main",
            "controller = cbf_pcc.controller:main",
            "field_cbf_optimizer = cbf_pcc.field_cbf_optimizer:main",
            "phi_marker_visualizer = cbf_pcc.phi_marker_visualizer:main",
            "phi_pointcloud_visualizer = cbf_pcc.phi_pointcloud_visualizer:main",
            "pose_collector = cbf_pcc.pose_collector:main",
            "sensing_region_marker_visualizer = cbf_pcc.sensing_region_marker_visualizer:main",
            "sensing_region_pointcloud_visualizer = cbf_pcc.sensing_region_pointcloud_visualizer:main",
        ],
    },
)
