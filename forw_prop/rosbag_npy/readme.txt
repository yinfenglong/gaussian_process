####### gazebo datas(.npy)##########
exp_data.npy is from folder "points_20_rosbag_m_1.709" (Odometry.twist.x,y,z)

exp_data_v_est.npy is from folder "points_20_rosbag_m_1.709_v_est" (function:velocity estimation. with LPF)

##############################
exp_data_pose_control_mpc.npy is from rosbag "2021-08-19-21-45-30.bag" with "export_mission_state_control_npy_mpc_get_x_next.py" and "/acados/quadrotor_optimizer_q_mpc_get_x_next.py" for comparing MPC: x_k+1 and forward_propgation x_k+1 (compare_mpc_forw_npy.py)

result in folder "figures_mpc_forw"

####### code #############
export_mission_state_control_npy_mpc_get_x_next.py
	Odometry.twist.x,y,z
	get mpc: x_next
	control_offset:0.59 --- verify u[3]!!!!
	launch: mpc_acados.cpp: control_offset:0.59

	this code doesn't work
