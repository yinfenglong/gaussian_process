 self.att_rate_cmd = np.array(
            [msg.body_rate.x, msg.body_rate.y, msg.body_rate.z, (msg.thrust / 0.59)*9.8066])

gazebo
20 random points

mpc_acados_q.cpp, control_offset = 0.59 and hier without velocity estimation!!!!!!!!
record datas in mission-loop from rosbag
2021-08-19-21-45-30.bag
mass:1.709 kg (plus 50g)
