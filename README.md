# coverage_control

## Requirements
- Ubuntu22.04
- ROS2 Humble
- Python3.10

## Installation
```sh
cd ~/ros2_ws/src/coverage_control
python3 -m pip install -r requirements/lib.txt
rosdep install -i -y --from-paths .
```

```sh
sudo python -m pip install -r requirements/tools.txt
```

[cbfpy/installation](https://github.com/toshi67026/cbfpy#installation)

## Usage
### simple coverage control
You can select the following parameter through the launch argument:
- dimension of the field: {1,2,3}
- number of agents: {1,2,3,4,5}
- initial density function(phi) type: {1: Uniform, 2: Gaussian, 3: Disk}

```sh
ros2 launch coverage_control scc.launch dim:={1,2,3} num:={1,2,3,4,5} phi:={1,2,3}
```

#### rviz
By adjusting the launch files and the checkboxes in the rviz panels, the density map(phi) and the sensing region(sr) of each agent can be drawn as `Marker` and `Pointcloud2` respectively, as shown below.

<img src=assets/scc_1d_3.png width=50%><img src=assets/scc_1d_5_phi.png width=50%>
<img src=assets/scc_2d_5.png width=50%><img src=assets/scc_2d_5_phi.png width=50%>
<img src=assets/scc_3d_5.png width=50%><img src=assets/scc_3d_4_phi.png width=50%>

#### rqt_graph
<img src=assets/scc_2d_rosgraph.png width=100%>

### persistent coverage control(pcc-devel branch)
```sh
ros2 launch coverage_control pcc.launch dim:={1,2,3} agent_num:={1,2,3,4,5}
```

#### rviz
<img src=assets/pcc_2d_5.gif width=70%>
<img src=assets/pcc_3d_5.gif width=70%>

## Tools
### Format
- isort
- black
```sh
task fmt
```

### Lint
- black
- ruff
```sh
task lint
```

### mypy
- mypy
```sh
task mypy
```
