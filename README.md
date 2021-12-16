# gaussian_process
## simulation and experiments
### collect training data for GP
1. Source enviroument in linux computer.
'''
cd test_ma_ws/
source ros.zsh
cd src/itm
tmuxinator start -p sitl_gazebo.yml
'''
2. Start MPC controller in simulation.
'''
roslaunch itm_nonlinear_mpc px4_exp_node.launch is_simulation:=true mpc_and_gp:=false
'''
3. Let the quadrotor take off and record rosbag.
'''
rosbag record -a
rostopic pub /itm_quadrotor_control/user_command itm_mav_msgs/SetMission "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
command_idx: 0
mission_mode: 1"
'''
4. The quadrotor flies on a trajectory.
'''
itm roslaunch itm_nonlinear_mpc offboard_sim_trajectory_generator_launch.launch horizon:=1 traj_type_idx:=1 

itm rosrun itm_nonlinear_mpc offboard_sim_random_gazebo_trajectory_commander.py 

offboard:
rostopic pub /itm_quadrotor_control/user_command itm_mav_msgs/SetMission "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
command_idx: 0
mission_mode: 3"
'''
5. Land the quadrotor.
'''
rostopic pub /itm_quadrotor_control/user_command itm_mav_msgs/SetMission "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
command_idx: 0
mission_mode: 2"
'''
### GP training
### standard process of GP training. 
'''
cd ./scripts/gaussian_process/gpr/gp_process.yml
tmuxinator start -p gp_process.yml folder_name=q330/20211008_10_with_cargo offset=0.44
'''

### training and prediction of different GP models
1. exact GP (velocity -> acceleration)
'''
python3 gpr_GPyTorch_train.py q330/20211008_10_with_cargo data_for_gp_y.npz
python3 gpr_GPyTorch_predict.py q330/20211008_10_with_cargo data_for_gp_y.npz

'''
2. exact GP (velocity, altitude -> acceleration)
'''
python3 gpr_GPyTorch_train_2d.py q330/20211008_10_with_cargo data_for_gp_y.npz
python3 gpr_GPyTorch_predict_2d.py q330/20211008_10_with_cargo data_for_gp_y.npz
'''
3. approximate GP prediction(velocity -> acceleration)
'''
python3 gpr_GPyTorch_dataload_train.py q330/20211008_10_with_cargo data_for_gp_y.npz
python3 gpr_GPyTorch_dataload_predict.py q330/20211008_10_with_cargo data_for_gp_y.npz
'''
4. approximate GP training(velocity, altitude -> acceleration
'''
python3 gpr_GPyTorch_approximate_train_2d.py q330/20211008_10_with_cargo data_for_gp_y.npz
python3 gpr_GPyTorch_approximate_predict_2d.py q330/20211008_10_with_cargo data_for_gp_y.npz
'''

### GP-based MPC
'''
roslaunch itm_nonlinear_mpc px4_exp_node.launch is_simulation:=true mpc_and_gp:=true
'''
## mathematical simulation
'''
python3 ./ACADOS_temp_set_p/quad_sim_q.py
'''

<!-- ## SITL

Install ROS Kinetic according to the [documentation](http://wiki.ros.org/kinetic/Installation), then [create a Catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace).

directory DataSSD/YanLI/Yan_ROS_ws

Terminal 1:
to connect to localhost, use this URL:

```
$roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
```

Terminal 2:
to run SITL wrapped in ROS the ROS environment needs to be updated, then launch as usual:
```
$cd <Firmware_clone>
$DONT_RUN=1 make px4_sitl_default gazebo
$source ~/catkin_ws/devel/setup.bash    # (optional)
$source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default
$export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)
$export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/Tools/sitl_gazebo
$no_sim=1 make px4_sitl_default gazebo
```
Terminal 3:

```
$cd <Firmware_clone>
$source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default
$roslaunch gazebo_ros empty_world.launch world_name:=$(pwd)/Tools/sitl_gazebo/worlds/iris.world
```
Terminal 4:
```
$roslaunch itm_nonlinear_mpc itm_nonlinear_mpc_sim_sitl.launch
```
Terminal 5: publish set point position topic

```
$rostopic pub /itm_quadrotot_control/set_point_pos geometry_msgs/PoseStamped“header:
```
Terminal 6:
```
roslaunch itm_nonlinear_mpc itm_nonlinear_mpc_sim_offboard.launch
``` 
