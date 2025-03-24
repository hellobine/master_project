# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces
# import rospy
# from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
# from nav_msgs.msg import Odometry
# from quadrotor_msgs.msg import ControlCommand, TrajectoryPoint, Trajectory

# from gazebo_msgs.msg import ModelState
# from rotors_comm.msg import WindSpeed
# import threading
# from std_msgs.msg import Bool
# import random
# import torch

# class QuadrotorEnv(gym.Env):
#     def __init__(self, namespace="drone"):
#         super(QuadrotorEnv, self).__init__()
#         self.state_lock = threading.Lock()  # 立即初始化
    
#         self.namespace = namespace  # ROS 命名空间，用于区分不同无人机实例

#         desired_pos_map = {
#             "hummingbird0": [0,0,3],
#             "hummingbird1": [0,1,3],
#             "hummingbird2": [0,2,3],
#             "hummingbird3": [1,0,3],
#             "hummingbird4": [1,1,3],
#             "hummingbird5": [1,2,3],
#             "hummingbird6": [2,0,3],
#             "hummingbird7": [2,1,3],
#             "hummingbird8": [2,2,3],
#             "hummingbird9": [3,3,3],

#         }


#         self.mass = 0.68  # kg
#         self.gravity = 9.8066
#         self.min_thrust = 0 * self.mass * self.gravity
#         self.max_thrust =  4 * self.mass * self.gravity
#         self.max_angular_rate = 5.0

#        # 位置 (3), 旋转矩阵 (3x3=9), 线速度 (3), 上一个动作 (4) -> 共 19 维
#         self.state_dim = 17  
#         self.action_dim = 4  # [推力, ωx, ωy, ωz]

#         self.test_action=0
        

#         self.max_position_error = 5.0  # 米
#         self.max_velocity_error = 5.0  # 米
#         self.max_episode_steps = 128
#         self.step_count = 0
#         self.episode_reward = 0
#         self.device = torch.device("cpu")
#         self.observation_space = spaces.Box(
#             low=np.concatenate((
#                 np.full(3, -1),          # ratget pos relative 
#                 np.full(4, -1),          # orientaion
#                 np.full(3, -1),          # line_vel
#                 np.full(3, -1),          # ang_vel
#                 np.full(1, 0),           # last_action_thrust
#                 np.full(3, -1)           # last_action_angular
#             )),
#             high=np.concatenate((
#                 np.full(3, 1),          # ratget pos relative 
#                 np.full(4, 1),          # orientaion
#                 np.full(3, 1),          # line_vel
#                 np.full(3, 1),          # ang_vel
#                 np.full(1, 1),           # last_action_thrust
#                 np.full(3, 1)           # 上一步动作
#             )),
#             dtype=np.float64
#         )

#         self.action_space = spaces.Box(
#             low=np.array([0,-1,-1, -1], dtype=np.float64),
#             high=np.array([1, 1, 1, 1], dtype=np.float64),
#             dtype=np.float64
#         )

#         if not rospy.core.is_initialized():
#             rospy.init_node(f'{self.namespace}_quadrotor_sb3_env', anonymous=True)
        
    
#         # 控制频率
#         self.control_hz = 50.0
#         self.rate = rospy.Rate(self.control_hz)
#         self.current_state = np.zeros(self.state_dim, dtype=np.float32)

#         self.desired_state = np.array(
#             desired_pos_map[namespace] + [0, 0, 0, 1] + [0, 0, 0],
#             dtype=np.float64
#         )


#         self.prev_real_thrust = None
#         self.prev_real_bodyrates = None
#         self.prev_obs = None
        
#         self.prev_action_ = None
#         self.prev_position = [0,0,0]
#         self.angular_x = 0
#         self.angular_y = 0
#         self.angular_z = 0


#         self.stable_count = 0
#         self.stable_steps_needed = 5   # 连续满足条件多少步后判定成功
#         self.stable_pos_threshold = 0.2  # 位置误差小于该值
#         self.stable_vel_threshold = 0.5  # 速度误差小于该值
#         self.reference_trajectory_ = None  # 使用列表模拟队列



#         self.yaw_lambda = -10.0  # 偏航奖励中使用的超参数
#         self.reward_scales = {
#             "target": 100.0,
#             "smooth": -1e-2,
#             "yaw": 0.01,
#             "angular": -2e-4,
#             "crash": -1.0
#         }
        


#         # self.pnh_ = rospy.get_namespace()  # 这里用 ROS 的命名空间代替参数句柄
  
#         self.reference_state_ = None  # 例如一个包含 position 属性的对象，position 为 numpy 数组
#         self.kPositionJumpTolerance_ = 0.1  # 示例值，单位与 position 相同
#         self.first_time_in_new_state_=True

    
#         self.odom_sub = rospy.Subscriber(f'/{self.namespace}/ground_truth/odometry', Odometry, self.odom_callback)
#         self.cmd_pub = rospy.Publisher(f'/{self.namespace}/control_command', ControlCommand, queue_size=1)
#         self.arm_pub = rospy.Publisher(f'/{self.namespace}/bridge/arm', Bool, queue_size=1)
#         self.reference_state_sub = rospy.Subscriber('/autopilot/trajectory', Trajectory, self.reference_trajectory_callback)

        
#         self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)
   

#         rospy.sleep(1.0)
    
#         if not rospy.is_shutdown():
#             msg = Bool(data=True)
#             self.arm_pub.publish(msg)
#             rospy.loginfo("Published arm message: true")
#         self._reset_drone_pose()
    
#     def _quaternion_to_rotation_matrix(self, x, y, z, w):
#         """将四元数转换为旋转矩阵（3x3），注意四元数顺序为 (x, y, z, w)"""
#         R = np.array([
#             [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
#             [2*(x*y + z*w),           1 - 2*(x**2 + z**2),    2*(y*z - x*w)],
#             [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
#         ], dtype=np.float32)
#         return R
    

#     def odom_callback(self, msg):
#         # with self.state_lock:
#             # 提取位置信息，并转换为torch张量
#             pos = torch.tensor([
#                 msg.pose.pose.position.x,
#                 msg.pose.pose.position.y,
#                 msg.pose.pose.position.z
#             ], dtype=torch.float32, device=self.device)
            
#             # 从desired_state中获取目标位置，并转换为torch张量
#             target_pos = torch.tensor(self.desired_state[0:3], dtype=torch.float32, device=self.device)
            
#             # 按照观测配置中的缩放因子 1/3 对相对位置归一化，并clip到[-1, 1]
#             pos_rel = torch.clip((pos - target_pos) * (1 / self.max_position_error), -1, 1)
            
#             # 提取四元数（orientation），保持原始值
#             orientation = torch.tensor([
#                 msg.pose.pose.orientation.x,
#                 msg.pose.pose.orientation.y,
#                 msg.pose.pose.orientation.z,
#                 msg.pose.pose.orientation.w
#             ], dtype=torch.float32, device=self.device)
            
#             # 提取线速度，并按照缩放因子 1/3 归一化，clip到[-1, 1]
#             lin_vel = torch.tensor([
#                 msg.twist.twist.linear.x,
#                 msg.twist.twist.linear.y,
#                 msg.twist.twist.linear.z
#             ], dtype=torch.float32, device=self.device)
            
#             lin_vel = torch.clip(lin_vel * (1 / self.max_velocity_error), -1, 1)
            
#             # 提取角速度，并按照缩放因子 1/π 归一化，clip到[-1, 1]
#             ang_vel = torch.tensor([
#                 msg.twist.twist.angular.x,
#                 msg.twist.twist.angular.y,
#                 msg.twist.twist.angular.z
#             ], dtype=torch.float32, device=self.device)
#             ang_vel = torch.clip(ang_vel * (1 / self.max_angular_rate), -1, 1)
            
#             # 上一次的动作，如果为空则使用全零向量（保持原值，不归一化）
#             prev_action = self.prev_action_ if self.prev_action_ is not None else np.zeros(self.action_dim, dtype=np.float32)
#             prev_action = torch.tensor(prev_action, dtype=torch.float32, device=self.device)
            
#             # 构造当前状态：依次拼接归一化后的相对位置、四元数、归一化后的线速度、归一化后的角速度和上次动作
#             self.current_state = torch.cat([
#                 pos_rel, #3
#                 orientation,#4 
#                 lin_vel, #3
#                 ang_vel, #3
#                 prev_action #4
#             ], dim=0).cpu().numpy()

#             # print(self.current_state)



#     # def odom_callback(self, msg):
#     #     with self.state_lock:
#     #         pos = [msg.pose.pose.position.x, 
#     #                msg.pose.pose.position.y,
#     #                msg.pose.pose.position.z]

#     #         qx = msg.pose.pose.orientation.x
#     #         qy = msg.pose.pose.orientation.y
#     #         qz = msg.pose.pose.orientation.z
#     #         qw = msg.pose.pose.orientation.w
#     #         rot_mat = self._quaternion_to_rotation_matrix(qx, qy, qz, qw).flatten()

#     #         orientation = [qx,qy,qz,qw]

#     #         self.angular_x = msg.twist.twist.angular.x
#     #         self.angular_y = msg.twist.twist.angular.y
#     #         self.angular_z = msg.twist.twist.angular.z


#     #         lin_vel = [msg.twist.twist.linear.x,
#     #                    msg.twist.twist.linear.y,
#     #                    msg.twist.twist.linear.z]

#     #         prev_action = self.prev_action_ if self.prev_action_ is not None else np.zeros(self.action_dim, dtype=np.float32)
            
#     #         target_pos = self.desired_state[0:3]

#     #         pos_rel =  torch.clip((pos - target_pos)/3)

#     #         target_vel = self.desired_state[12:15]
#     #         target_ori = self.desired_state[3:12]
#     #         self.current_state = np.array(
#     #             pos_rel + orientation.tolist() + lin_vel  +
#     #             prev_action.tolist(), dtype=np.float32
#     #         )

#     def reference_trajectory_callback(self, msg):
#         print("---------------------------grhytgfdc-------------------------")
  

#     def step(self, action):
#         self.step_count += 1
#         self._publish_action(action)
#         self.rate.sleep()
        
#         with self.state_lock:
#             obs = self.current_state.copy()

#         # reward = self._compute_reward(obs, action)
#         reward = self._compute_reward3(obs, action)# / (1/self.control_hz)

#         check_done = self._check_done(obs)
#         # if check_done:
#         #     reward-=5

#         self.episode_reward += reward

#         if self.step_count >= self.max_episode_steps or check_done:
#             # if self.stable_count >= self.stable_steps_needed:
#             #     reward+= 256
#             #     self.episode_reward += reward

#             if self.step_count >= self.max_episode_steps:
#                 terminated = False  # 非任务失败
#                 truncated = True   # 由于达到最大步数
#             else:
#                 terminated = True  # 任务失败，如超出范围
#                 truncated = False
 
#             info = {"reward": reward, "episode": {"r": self.episode_reward, "l": self.step_count}}
#             # 重置计数器和累计奖励
#             self.episode_reward = 0
#             self.step_count = 0
#             self.stable_count = 0
#         else:
#             terminated = False
#             truncated = False
#             info = {"reward": reward}

#         # 重置计数器和累计奖励
#         if terminated or truncated:
#             self.episode_reward = 0
#             self.step_count = 0


#         return obs, reward, terminated, truncated, info

#     def reset(self, **kwargs):
#         self.step_count = 0
#         self.episode_reward = 0
#         self.prev_action_ = None
#         self._reset_drone_pose()
#         rospy.sleep(0.01)  # 等待复位完成
#         with self.state_lock:
#             obs = self.current_state.copy()
#         return obs, {}
    

#     def _reset_drone_pose(self):
#         init_position = np.array([
#             np.random.uniform(-1.0, 1.0),  # x 范围 [-1, 1] m
#             np.random.uniform(-1.0, 1.0),  # y 范围 [-1, 1] m
#             np.random.uniform(0, 3)    # z 范围 [0.5, 1.5] m，避免过高或过低
#         ])

#         init_orientation = np.random.uniform(-self.max_position_error, self.max_position_error, size=3)  # 近似水平的随机扰动
#         init_velocity = np.random.uniform(-0.3, 0.3, size=3)  # 低速随机初始化 #init_velocity = np.random.uniform(-3, 3, size=3)  # 低速随机初始化
#         init_angular_velocity = np.random.uniform(-0.2, 0.2, size=3)  # 低速角速度初始化

#         state = ModelState()
#         state.model_name = self.namespace  
#         state.pose.position.x = self.desired_state[0]
#         state.pose.position.y = self.desired_state[1]
#         state.pose.position.z = 0.11
#         state.pose.orientation.x = 0
#         state.pose.orientation.y = 0
#         state.pose.orientation.z = 0
#         state.pose.orientation.w = 1  

#         state.twist.linear.x=init_velocity[0]
#         state.twist.linear.y=init_velocity[1]
#         state.twist.linear.z=init_velocity[2]

#         state.twist.angular.x=init_angular_velocity[0]
#         state.twist.angular.y=init_angular_velocity[1]
#         state.twist.angular.z=init_angular_velocity[2]

#         pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
#         pub.publish(state)

#     def _publish_action(self, action):
#         control_cmd = ControlCommand()
#         control_cmd.header.stamp = rospy.Time.now()
#         control_cmd.expected_execution_time = rospy.Time.now()
#         control_cmd.armed = True
#         control_cmd.control_mode = ControlCommand.BODY_RATES

#         thrust = self.max_thrust * action[0] # ((action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
#         control_cmd.collective_thrust = thrust

#         bodyrates = np.clip(action[1:], -1, 1) * self.max_angular_rate

        
#         control_cmd.bodyrates.x = bodyrates[0]
#         control_cmd.bodyrates.y = bodyrates[1]
#         control_cmd.bodyrates.z = bodyrates[2]
#         # print(control_cmd.header.stamp)
#         self.cmd_pub.publish(control_cmd)


#     def _quaternion_to_euler(self, x, y, z, w):
#         """将四元数转换为欧拉角 (roll, pitch, yaw)"""
#         t0 = 2.0 * (w * x + y * z)
#         t1 = 1.0 - 2.0 * (x * x + y * y)
#         roll = np.arctan2(t0, t1)
        
#         t2 = 2.0 * (w * y - z * x)
#         t2 = np.clip(t2, -1.0, 1.0)
#         pitch = np.arcsin(t2)
        
#         t3 = 2.0 * (w * z + x * y)
#         t4 = 1.0 - 2.0 * (y * y + z * z)
#         yaw = np.arctan2(t3, t4)
#         return roll, pitch, yaw

#     def _compute_reward1(self, obs, curr_action):
#         curr_pos = obs[0:3]
#         target_pos = self.desired_state[0:3]

#         curr_vel = obs[12:15]
#         target_vel = self.desired_state[12:15]

#         prev_dist = np.linalg.norm(target_pos - self.prev_position)
#         curr_dist = np.linalg.norm(target_pos - curr_pos)
#         r_progress_dis = prev_dist - curr_dist

#         r_progress = np.tanh(0.8*r_progress_dis)

#         # 1. 位置跟踪奖励（指数衰减）
#         pos_error = np.linalg.norm(curr_pos - target_pos)
#         r_position = np.exp(-1 * pos_error)

#         # if(r_position<0.8):
#         #     r_progress=0

#         # if self.stable_count >= self.stable_steps_needed:
#             # r_position+= 5
#             # r_progress=0

#         # 2. 速度跟踪奖励
#         vel_error = np.linalg.norm(curr_vel - target_vel)
#         r_velocity = np.exp(-0.3 * vel_error)

#         # 3. 姿态稳定奖励（旋转矩阵与目标差异）
#         current_rot = obs[3:12].reshape(3,3)
#         target_rot = self.desired_state[3:12].reshape(3,3)
#         rot_diff = np.arccos((np.trace(current_rot @ target_rot.T) - 1)/2)
#         # r_attitude = (1 - np.abs(rot_diff/np.pi))
#         # r_attitude = np.abs(rot_diff/np.pi)
#         r_attitude = np.exp(-0.3 * np.abs(rot_diff))

#         curr_real_thrust = self.max_thrust * curr_action[0] # ((curr_action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
#         curr_real_bodyrates = np.clip(curr_action[1:], -1, 1) * self.max_angular_rate

#         # 4. 动作平滑惩罚
#         if self.prev_action_ is not None:
#             real_diff_thrust = curr_real_thrust - self.prev_real_thrust
#             real_diff_bodyrates = curr_real_bodyrates - self.prev_real_bodyrates
#             real_action_diff = np.linalg.norm(
#             np.concatenate(([real_diff_thrust], real_diff_bodyrates))
#             )
#             # r_smooth = -0.3 * (real_action_diff ** 2)
#             action_diff = np.linalg.norm(curr_action - self.prev_action_)
#             r_smooth =  np.exp(-0.3 * action_diff)
#         else:
#             r_smooth = 0.0
       
#         # 5. 能量效率惩罚
#         thrust = ((curr_action[0] + 1)/2) * (self.max_thrust - self.min_thrust) + self.min_thrust
#         r_energy = -0.1 * (thrust/(self.mass*self.gravity))**2

#         # if pos_error<0.5:
#         #     r_progress_weight = 0.6
#         #     r_position_weight = 2
#         #     r_velocity_weight = 1 
#         #     r_attitude_weight = 1
#         #     r_smooth_weight = 0.3
#         # else:

#         r_progress_weight = 0.0
#         r_position_weight = 1
#         r_velocity_weight = 0.2
#         r_attitude_weight = 0.3
#         r_smooth_weight = 0.6

#         # print("r_progress: ", r_progress_weight*r_progress, r_position_weight*r_position, r_velocity_weight*r_velocity, r_attitude_weight*r_attitude, r_smooth_weight*r_smooth)
        
#         total_reward = (
#             r_progress_weight*r_progress +
#             r_position_weight*r_position +
#             r_velocity_weight*r_velocity +
#             r_attitude_weight*r_attitude +
#             r_smooth_weight*r_smooth 
#             # +
#             # r_energy
#         )

#         pos_threshold = 0.3    # 位置误差小于0.3米
#         vel_threshold = 0.5    # 速度误差小于0.5m/s
#         rot_threshold = 0.3    # 姿态误差(弧度)小于0.3 (约17.2度)

#         # 如果都满足，就额外 +2
#         # if pos_error < pos_threshold and vel_error < vel_threshold and abs(rot_diff) < rot_threshold:
#         #     print("hovering !!!!!!!!!!!!!!!!!!!!!!!!")
#         #     total_reward += 6.0
#         # elif pos_error < pos_threshold*0.8 and vel_error < vel_threshold*0.8 and abs(rot_diff) < rot_threshold*0.8:
#         #     print("greater hovering !!!!!!!!!!!!!!!!!!!!!!!!")
#         #     total_reward += 12.0
#         #         # 当误差更严格时，给予更高的奖励
#         if pos_error < pos_threshold and vel_error < vel_threshold and abs(rot_diff) < rot_threshold:
#             # 计算位置、速度和姿态误差比例（误差越小比例越接近1）
#             pos_factor = 1 - pos_error / pos_threshold
#             vel_factor = 1 - vel_error / vel_threshold
#             rot_factor = 1 - abs(rot_diff) / rot_threshold
#             # 三项平均后再乘以系数（最大奖励12.0分）
#             hover_reward = 24.0 * (pos_factor + vel_factor + rot_factor) / 3.0
#             print("Dynamic hovering reward: ", hover_reward)
#             total_reward += hover_reward

#         # print("pos_error: ",pos_error , ", vel_error: ",vel_error, ",  abs(rot_diff): ", abs(rot_diff))
        
#         # print("total_reward: ", total_reward, ", r_progress: ", r_progress_weight*r_progress, r_position_weight*r_position, r_velocity_weight*r_velocity, r_attitude_weight*r_attitude, r_smooth_weight*r_smooth)
        
#         # 更新历史数据
#         self.prev_action_ = curr_action.copy()
#         self.prev_real_thrust = curr_real_thrust.copy()
#         self.prev_real_bodyrates = curr_real_bodyrates.copy()

#         self.prev_position = curr_pos.copy()
#         return total_reward

#     def _compute_reward(self, obs, curr_action):
#         # 提取当前状态和目标状态
#         curr_pos = obs[0:3]
#         target_pos = self.desired_state[0:3]
#         curr_vel = obs[12:15]
#         target_vel = self.desired_state[12:15]
        
#         # 定义归一化的误差阈值（悬停误差阈值）
#         pos_threshold = self.max_position_error    # 位置误差阈值（米）
#         vel_threshold = self.max_velocity_error    # 速度误差阈值（m/s）
#         rot_threshold = 1    # 姿态误差阈值（弧度）
        
#         # 1. 位置误差（平方差归一化）
#         pos_error = np.linalg.norm(curr_pos - target_pos)
#         pos_error_sq = pos_error ** 2
#         # 归一化：当 pos_error == pos_threshold 时比例为1
#         r_position = np.clip(1 - pos_error_sq / (pos_threshold ** 2), 0, 1)
        
#         # 2. 速度误差（平方差归一化）
#         vel_error = np.linalg.norm(curr_vel - target_vel)
#         vel_error_sq = vel_error ** 2
#         r_velocity = np.clip(1 - vel_error_sq / (vel_threshold ** 2), 0, 1)
        
#         # 3. 姿态误差（使用旋转矩阵的误差，转换为旋转角误差）
#         current_rot = obs[3:12].reshape(3, 3)
#         target_rot = self.desired_state[3:12].reshape(3, 3)
#         # 计算旋转矩阵之间的夹角，注意数值稳定性（clip到 [-1, 1]）
#         dot_val = np.clip((np.trace(current_rot @ target_rot.T) - 1) / 2, -1.0, 1.0)
#         rot_diff = np.arccos(dot_val)
#         rot_error_sq = (abs(rot_diff)) ** 2
#         # print("rot_error_sq: ", rot_error_sq)
#         r_attitude = np.clip(1 - rot_error_sq / (rot_threshold ** 2), 0, 1)
        
#         # 4. 动作平滑（使用当前动作与上一次动作的平方差归一化）
#         if self.prev_action_ is not None:
#             action_diff = np.linalg.norm(curr_action - self.prev_action_)
#             action_diff_sq = action_diff ** 2
#             # print("action_diff_sq: ", action_diff_sq)
#             # 假设1.0为一个合理的动作差值上限，当差值超过1则 r_smooth 为0
#             r_smooth = np.clip(1 - action_diff_sq / 4.0, 0, 1)
#         else:
#             r_smooth = 1.0  # 初始步默认平滑
        
#         # 5. 能量效率：鼓励使用较低推力（归一化推力后采用平方形式）
#         # 计算当前推力（假设动作[0] 范围 [0, 1] 映射到 [min_thrust, max_thrust]）
#         thrust = self.max_thrust * curr_action[0]
#         norm_thrust = (thrust - self.min_thrust) / (self.max_thrust - self.min_thrust)  # 范围 [0, 1]
#         # 能量奖励：推力越低越好，这里采用1 - (normalized_thrust)^2
#         r_energy = np.clip(1 - norm_thrust ** 2, 0, 1)
        
#         # 各项奖励按权重组合
#         # 例如：位置 0.4，速度 0.2，姿态 0.2，动作平滑 0.1，能量 0.1
#         total_reward = (
#             1 * r_position 
#             # 0.3 * r_velocity +
#             # 0.3 * r_attitude +
#             # 0.4 * r_smooth 
#             # 0.1 * r_energy
#         )
#         # print("total_reward: ", total_reward)

#         pos_threshold = 0.6    # 位置误差小于0.3米
#         vel_threshold = 0.8    # 速度误差小于0.5m/s
#         rot_threshold = 0.8    # 姿态误差(弧度)小于0.3 (约17.2度)

#         if pos_error < pos_threshold and vel_error < vel_threshold and abs(rot_diff) < rot_threshold:
#         # if pos_error < pos_threshold or vel_error < vel_threshold or abs(rot_diff) < rot_threshold :
#             # 计算位置、速度和姿态误差比例（误差越小比例越接近1）
#             pos_factor = 1 - pos_error / pos_threshold
#             vel_factor = 1 - vel_error / vel_threshold
#             rot_factor = 1 - abs(rot_diff) / rot_threshold
#             # 三项平均后再乘以系数（最大奖励12.0分）
#             hover_reward = 24.0 * (pos_factor + vel_factor + rot_factor) / 3.0
#             # hover_reward = 6
#             print("Dynamic hovering reward: ", hover_reward)
#             total_reward += hover_reward
#         # print(pos_error, vel_error, abs(rot_diff))
        
#         # 输出各项奖励以便调试（可根据需要删除）
#         # print("r_position: {:.3f}, r_velocity: {:.3f}, r_attitude: {:.3f}, r_smooth: {:.3f}, r_energy: {:.3f}".format(
#         #     r_position, r_velocity, r_attitude, r_smooth, r_energy))
#         # print("total_reward: {:.3f}".format(total_reward))
        
#         # 更新历史数据：记录当前动作和位置
#         self.prev_action_ = curr_action.copy()
#         self.prev_position = curr_pos.copy()
        
#         return total_reward
    

#     def _compute_reward3(self, obs, curr_action):
#         """
#         根据观测 obs（17维）和当前动作 curr_action 计算总奖励，奖励拆分为：
#         1. 目标奖励：利用上一次和当前相对目标位置（归一化后还原）距离平方的差值，
#             正值表示向目标靠近。
#         2. 平滑奖励：当前动作与上一次动作的差值平方和（大变化惩罚）。
#         3. 偏航奖励：从观测中的四元数计算 yaw 角，归一化到 [-π, π] 后，
#             用指数函数计算奖励（超参数 yaw_lambda 控制敏感度）。
#         4. 角速度奖励：直接使用观测中归一化后角速度的范数。
#         5. 撞击奖励：若发生撞击或失稳，则给一个固定惩罚。
        
#         （注意：观测空间各部分含义如下：
#         - obs[0:3]：目标相对位置（归一化至[-1,1]，实际单位需要乘以3）
#         - obs[3:7]：四元数（orientation）
#         - obs[7:10]：线速度（归一化）
#         - obs[10:13]：角速度（归一化，即除以 π）
#         - obs[13:14]：上一次的推力动作
#         - obs[14:17]：上一次的角速度动作
#         )
#         """
#         # 定义归一化还原因子
#         target_scale = 3.0

#         # 1. 目标奖励：利用归一化后的目标相对位置还原为实际单位
#         curr_rel = obs[0:3] * self.max_position_error
#         if self.prev_obs is not None:
#             last_rel = self.prev_obs[0:3] * self.max_position_error
#         else:
#             last_rel = curr_rel.copy()
#         r_target = np.sum(last_rel**2) - np.sum(curr_rel**2)

#         # 2. 平滑奖励：当前动作与上一次动作的差值平方和
#         if self.prev_action_ is not None:
#             r_smooth = np.sum(np.square(curr_action - self.prev_action_))
#         else:
#             r_smooth = 0.0

#         # 3. 偏航奖励：从观测中的四元数计算 yaw 角
#         # 观测中四元数存放在 obs[3:7]，假定顺序为 [x, y, z, w]
#         quat = obs[3:7]
#         x, y, z, w = quat
#         # 使用常用公式计算 yaw（弧度）
#         yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
#         # 归一化到 [-π, π]
#         yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
#         # 使用超参数 yaw_lambda 计算指数奖励（例如 yaw_lambda = -10.0）
#         yaw_lambda = -10.0
#         r_yaw = np.exp(yaw_lambda * np.abs(yaw))

#         # 4. 角速度奖励：使用归一化后的角速度（obs[10:13]，原始角速度/π）的范数
#         r_angular = np.linalg.norm(obs[10:13])

#         # 5. 撞击奖励：若发生撞击或失稳，则给固定惩罚
#         r_crash = 1.0 if self._check_done(obs) else 0.0

#         # 各奖励的缩放系数在初始化时已经设置，并乘以了时间步 dt，这里直接相加

#         r_target_contrib = self.reward_scales["target"] * r_target
#         r_smooth_contrib = self.reward_scales["smooth"] * r_smooth
#         r_yaw_contrib = self.reward_scales["yaw"] * r_yaw
#         r_angular_contrib = self.reward_scales["angular"] * r_angular
#         r_crash_contrib = self.reward_scales["crash"] * r_crash

#         # total_reward = (r_target_contrib + r_smooth_contrib +
#         #                 r_yaw_contrib + r_angular_contrib +
#         #                 r_crash_contrib)

#         total_reward = (r_target_contrib)
#         # print("r_target contribution:", r_target_contrib)
#         # print("r_smooth contribution:", r_smooth_contrib)
#         # print("r_yaw contribution:", r_yaw_contrib)
#         # print("r_angular contribution:", r_angular_contrib)
#         # print("r_crash contribution:", r_crash_contrib)
#         # print("Total Reward:", total_reward)

#         # 更新历史记录
#         self.prev_obs = obs.copy()
#         self.prev_action_ = curr_action.copy()

#         return total_reward



#     def _check_done(self, obs):
#         # 位置失稳检查
#         pos_error = np.linalg.norm(obs[0:3]*self.max_position_error)
#         if pos_error > self.max_position_error:
#             # print("pos_error: ",pos_error)
#             return True
#         # if obs[0] > self.max_position_error or obs[0]<-self.max_position_error \
#         #     or obs[1] > self.max_position_error or obs[1]<-self.max_position_error \
#         #     or obs[2] > self.max_position_error or pos_error > self.max_position_error or obs[2]<0:
#         #     return True
        
#         # 姿态异常检查
#             # 姿态检查：检测侧翻（roll 超过阈值）
#         # current_rot = obs[3:12].reshape(3,3)
#         # # # 直接从旋转矩阵中计算 roll: roll = arctan2(R[2,1], R[2,2])
#         # roll = np.arctan2(current_rot[2,1], current_rot[2,2])
#         # if np.abs(roll) > 1.8:  # 阈值 1.0 弧度（约57°），可根据需求调整
#         #     # print("np.abs(roll) > 1.0")
#         #     return True

#         quaternion = obs[3:7]
#         x, y, z, w = quaternion

#         # 从四元数计算 roll 角
#         sinr_cosp = 2.0 * (w * x + y * z)
#         cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
#         roll = np.arctan2(sinr_cosp, cosr_cosp)

#         if np.abs(roll) > 1.8:  # 阈值：1.8 弧度
#             # print("roll")
#             return True

#         # 一旦满足 stable_steps_needed，则说明已经稳定悬停，done = True
#         # if self.stable_count >= self.stable_steps_needed:
#             # print("Quadrotor hovered successfully!")
#             # return True

#         # # 速度失控检查
#         if (np.linalg.norm(obs[7:10]*self.max_velocity_error)) > self.max_velocity_error:  # 最大允许速度
#         #     print("np.linalg.norm(obs[12:15]) > ", self.max_velocity_error)
#             # print("speed")
#             return True
        
#         return False
    
#     def render(self, mode='human'):
#         pass  # Gazebo 自带可视化

#     def close(self):
#         rospy.signal_shutdown("Environment closed")




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand, TrajectoryPoint, Trajectory

from gazebo_msgs.msg import ModelState
from rotors_comm.msg import WindSpeed
import threading
from std_msgs.msg import Bool
import random

class QuadrotorEnv(gym.Env):
    def __init__(self, namespace="drone"):
        super(QuadrotorEnv, self).__init__()
        self.state_lock = threading.Lock()  # 立即初始化
    
        self.namespace = namespace  # ROS 命名空间，用于区分不同无人机实例
        desired_pos_map = {
            "hummingbird0": [0,0,3],
            "hummingbird1": [0,1,3],
            "hummingbird2": [0,2,3],
            "hummingbird3": [1,0,3],
            "hummingbird4": [1,1,3],
            "hummingbird5": [1,2,3],
            "hummingbird6": [2,0,3],
            "hummingbird7": [2,1,3],
            "hummingbird8": [2,2,3],
            "hummingbird9": [3,3,3],

        }


        self.mass = 0.68  # kg
        self.gravity = 9.8066
        self.min_thrust = 1 * self.mass * self.gravity
        self.max_thrust =  3 * self.mass * self.gravity
        self.max_angular_rate = 3.0

       # 位置 (3), 旋转矩阵 (3x3=9), 线速度 (3), 上一个动作 (4) -> 共 19 维
        self.state_dim = 25  
        self.action_dim = 4  # [推力, ωx, ωy, ωz]
        

        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.full(3, -5),          # 当前位置
                np.full(9, -1),          # 旋转矩阵
                np.full(3, -5),          # 当前线速度
                np.full(3, -5),          # 目标位置
                np.full(3, -5),          # 目标速度
                np.full(4, -1)           # 上一步动作
            )),
            high=np.concatenate((
                np.full(3, 5),
                np.full(9, 1),
                np.full(3, 5),
                np.full(3, 5),
                np.full(3, 5),
                np.full(4, 1)
            )),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1]*self.action_dim, dtype=np.float32),
            high=np.array([1]*self.action_dim, dtype=np.float32),
            dtype=np.float32
        )

        if not rospy.core.is_initialized():
            rospy.init_node(f'{self.namespace}_quadrotor_sb3_env', anonymous=True)
        
    
        # 控制频率
        self.control_hz = 50.0
        self.rate = rospy.Rate(self.control_hz)
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.desired_state = np.array(
            desired_pos_map[namespace] + [1, 0, 0, 0, 1, 0, 0, 0, 1] + [0, 0, 0],
            dtype=np.float32
        )

        self.max_position_error = 3.0  # 米
        self.max_episode_steps = 256
        self.step_count = 0
        self.episode_reward = 0

        self.prev_real_thrust = None
        self.prev_real_bodyrates = None

        
        self.prev_action_ = None
        self.prev_position = [0,0,0]


        self.angular_x = 0
        self.angular_y = 0
        self.angular_z = 0

        self.stable_count = 0
        self.stable_steps_needed = 5   # 连续满足条件多少步后判定成功
        self.stable_pos_threshold = 0.2  # 位置误差小于该值
        self.stable_vel_threshold = 0.3  # 速度误差小于该值
        self.reference_trajectory_ = None  # 使用列表模拟队列


        # self.pnh_ = rospy.get_namespace()  # 这里用 ROS 的命名空间代替参数句柄
  
        self.reference_state_ = None  # 例如一个包含 position 属性的对象，position 为 numpy 数组
        self.kPositionJumpTolerance_ = 0.1  # 示例值，单位与 position 相同
        self.first_time_in_new_state_=True

    
        self.odom_sub = rospy.Subscriber(f'/{self.namespace}/ground_truth/odometry', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher(f'/{self.namespace}/control_command', ControlCommand, queue_size=1)
        self.arm_pub = rospy.Publisher(f'/{self.namespace}/bridge/arm', Bool, queue_size=1)
        self.reference_state_sub = rospy.Subscriber('/autopilot/trajectory', Trajectory, self.reference_trajectory_callback)

        
        self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)
   

        rospy.sleep(1.0)
    
        if not rospy.is_shutdown():
            msg = Bool(data=True)
            self.arm_pub.publish(msg)
            rospy.loginfo("Published arm message: true")
    
    
    def _quaternion_to_rotation_matrix(self, x, y, z, w):
        """将四元数转换为旋转矩阵（3x3），注意四元数顺序为 (x, y, z, w)"""
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2),    2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        return R
        
    def odom_callback(self, msg):
        if self.reference_trajectory_ is not None:
            current_time = rospy.Time.now()
            if self.first_time_in_new_state_:
                self.first_time_in_new_state_ = False
                self.time_start_trajectory_execution_ = current_time
            
            dt = current_time - self.time_start_trajectory_execution_
            dt_sec = dt.to_sec()

            ref_point = None
            for point in self.reference_trajectory_.points:
                point_time = point.time_from_start.to_sec()
                if point_time >= dt_sec:
                    ref_point = point
                    break

            # 如果 dt 超出所有轨迹点时间，则选取最后一个点
            if ref_point == None:
                # ref_point = self.reference_trajectory_.points[-1]
                self.first_time_in_new_state_=True


            pos = [ref_point.pose.position.x, 
                ref_point.pose.position.y,
                ref_point.pose.position.z]

            qx = ref_point.pose.orientation.x
            qy = ref_point.pose.orientation.y
            qz = ref_point.pose.orientation.z
            qw = ref_point.pose.orientation.w

            rot_mat = self._quaternion_to_rotation_matrix(qx, qy, qz, qw).flatten()

            lin_vel = [ref_point.velocity.linear.x,
                    ref_point.velocity.linear.y,
                    ref_point.velocity.linear.z]
            
            self.desired_state = np.array(pos + rot_mat.tolist() + lin_vel , dtype=np.float32)

        with self.state_lock:
            pos = [msg.pose.pose.position.x, 
                   msg.pose.pose.position.y,
                   msg.pose.pose.position.z]

            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            rot_mat = self._quaternion_to_rotation_matrix(qx, qy, qz, qw).flatten()

            lin_vel = [msg.twist.twist.linear.x,
                       msg.twist.twist.linear.y,
                       msg.twist.twist.linear.z]

            prev_action = self.prev_action_ if self.prev_action_ is not None else np.zeros(self.action_dim, dtype=np.float32)
            
            target_pos = self.desired_state[0:3]
            target_vel = self.desired_state[12:15]
            target_ori = self.desired_state[3:12]
            self.current_state = np.array(
                pos + rot_mat.tolist() + lin_vel + 
                target_pos.tolist() + target_vel.tolist() +
                prev_action.tolist(), dtype=np.float32
            )
        # print(self.reference_trajectory_)
        

    def reference_trajectory_callback(self, msg):
        # print(msg)
        # if self.reference_trajectory_ == None:
        #     self.reference_trajectory_ = msg
        print("---------------------------grhytgfdc-------------------------")
  

    def step(self, action):
        self.step_count += 1
        self._publish_action(action)
        self.rate.sleep()
        
        with self.state_lock:
            obs = self.current_state.copy()

        reward = self._compute_reward(obs, action)
        self.episode_reward += reward
        # if self.stable_count >= self.stable_steps_needed:
        #     reward+= self.stable_steps_needed * 1
            # self.episode_reward += reward

        if self.step_count >= self.max_episode_steps or self._check_done(obs):
            # 如果达到最大步数，则标记为截断，否则标记为终止
            if self.step_count >= self.max_episode_steps:
                terminated = False  # 非任务失败
                truncated = True   # 由于达到最大步数
            else:
                terminated = True  # 任务失败，如超出范围
                truncated = False
 
            # 在所有结束条件下，都返回 episode 信息
            info = {"reward": reward, "episode": {"r": self.episode_reward, "l": self.step_count}}
            # 重置计数器和累计奖励
            self.episode_reward = 0
            self.step_count = 0
            self.stable_count = 0

        else:
            terminated = False
            truncated = False
            info = {"reward": reward}

        # 重置计数器和累计奖励
        if terminated or truncated:
            self.episode_reward = 0
            self.step_count = 0


        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.episode_reward = 0
        self.prev_action_ = None
        self._reset_drone_pose()
        rospy.sleep(0.01)  # 等待复位完成
        with self.state_lock:
            obs = self.current_state.copy()
        return obs, {}
    

    def _reset_drone_pose(self):
        init_position = np.array([
            np.random.uniform(-1.0, 1.0),  # x 范围 [-1, 1] m
            np.random.uniform(-1.0, 1.0),  # y 范围 [-1, 1] m
            np.random.uniform(0, 4)    # z 范围 [0.5, 1.5] m，避免过高或过低
        ])

        init_orientation = np.random.uniform(-0.4, 0.4, size=3)  # 近似水平的随机扰动
        init_velocity = np.random.uniform(-1, 1, size=3)  # 低速随机初始化 #init_velocity = np.random.uniform(-3, 3, size=3)  # 低速随机初始化
        init_angular_velocity = np.random.uniform(-0.1, 0.1, size=3)  # 低速角速度初始化

        state = ModelState()
        state.model_name = self.namespace  # 确保命名空间匹配 Gazebo 的无人机模型
        state.pose.position.x = self.desired_state[0]
        state.pose.position.y = self.desired_state[1]
        state.pose.position.z = 0.0
        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = 0
        state.pose.orientation.w = 1.0  # 保持单位四元数

        state.twist.linear.x=init_velocity[0]
        state.twist.linear.y=init_velocity[1]
        state.twist.linear.z=init_velocity[2]

        state.twist.angular.x=init_angular_velocity[0]
        state.twist.angular.y=init_angular_velocity[1]
        state.twist.angular.z=init_angular_velocity[2]

        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        pub.publish(state)

    def _publish_action(self, action):
        # print(action)
        cmd = ControlCommand()
        cmd.armed = True
        cmd.control_mode = ControlCommand.BODY_RATES
        # cmd.collective_thrust = float(np.clip(action[0], self.min_thrust, self.max_thrust))
        # cmd.collective_thrust = action[0]

        thrust = ((action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        cmd.collective_thrust = thrust

        # 角速度归一化映射到 [-max_angular_rate, max_angular_rate]
        bodyrates = np.clip(action[1:], -1, 1) * self.max_angular_rate
        # 分别赋值给 x, y, z
        cmd.bodyrates.x = bodyrates[0]
        cmd.bodyrates.y = bodyrates[1]
        cmd.bodyrates.z = bodyrates[2]

        # self.prev_action_ = []

        self.cmd_pub.publish(cmd)


    def _quaternion_to_euler(self, x, y, z, w):
        """将四元数转换为欧拉角 (roll, pitch, yaw)"""
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return roll, pitch, yaw

    def _compute_reward(self, obs, curr_action):


        curr_pos = obs[0:3]
        target_pos = self.desired_state[0:3]



        curr_vel = obs[12:15]
        target_vel = self.desired_state[12:15]

        # print(self.desired_state)

        prev_dist = np.linalg.norm(target_pos - self.prev_position)
        curr_dist = np.linalg.norm(target_pos - curr_pos)
        r_progress_dis = prev_dist - curr_dist

        r_progress = np.tanh(2*r_progress_dis)
        # print(r_progress)


        # 1. 位置跟踪奖励（指数衰减）
        pos_error = np.linalg.norm(curr_pos - target_pos)
        r_position = np.exp(-0.5 * pos_error)

        if(r_position<0.8):
            r_progress=0

        if self.stable_count >= self.stable_steps_needed:
            r_position+= self.stable_steps_needed * 1
            r_progress=0

        # 2. 速度跟踪奖励
        vel_error = np.linalg.norm(curr_vel - target_vel)
        r_velocity = 0.5 * np.exp(-2 * vel_error)

        # 3. 姿态稳定奖励（旋转矩阵与目标差异）
        current_rot = obs[3:12].reshape(3,3)
        target_rot = self.desired_state[3:12].reshape(3,3)
        rot_diff = np.arccos((np.trace(current_rot @ target_rot.T) - 1)/2)
        r_attitude = 0.3 * (1 - np.abs(rot_diff/np.pi))


        curr_real_thrust = ((curr_action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        curr_real_bodyrates = np.clip(curr_action[1:], -1, 1) * self.max_angular_rate


        # 4. 动作平滑惩罚
        if self.prev_action_ is not None:
            real_diff_thrust = curr_real_thrust - self.prev_real_thrust
            real_diff_bodyrates = curr_real_bodyrates - self.prev_real_bodyrates
            real_action_diff = np.linalg.norm(
            np.concatenate(([real_diff_thrust], real_diff_bodyrates))
            )
            r_smooth = -0.3 * (real_action_diff ** 2)
            action_diff = np.linalg.norm(curr_action - self.prev_action_)


            r_smooth = -0.1 * (action_diff ** 2)
        else:
            r_smooth = 0.0

        # 5. 能量效率惩罚
        thrust = ((curr_action[0] + 1)/2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        r_energy = -0.05 * (thrust/(self.mass*self.gravity))**2

        # print(r_progress, r_position, r_velocity, r_attitude, r_smooth)
        # 合成总奖励
        total_reward = (
            r_progress +
            10*r_position +
            r_velocity +
            r_attitude +
            r_smooth
              +
            r_energy
        )

        # 更新历史数据
        self.prev_action_ = curr_action.copy()
        self.prev_real_thrust = curr_real_thrust.copy()
        self.prev_real_bodyrates = curr_real_bodyrates.copy()

        self.prev_position = curr_pos.copy()
        return total_reward
        
    def _check_done(self, obs):
        # 位置失稳检查
        pos_error = np.linalg.norm(obs[0:3] - self.desired_state[0:3])
        if obs[0] > self.max_position_error or obs[0]<-self.max_position_error \
            or obs[1] > self.max_position_error or obs[1]<-self.max_position_error \
            or obs[2] > 4 or obs[2] < 0:
            # print(obs)
        # if pos_error > self.max_position_error or obs[2] < 0:
            return True
        
        # 姿态异常检查
            # 姿态检查：检测侧翻（roll 超过阈值）
        # current_rot = obs[3:12].reshape(3,3)
        # # 直接从旋转矩阵中计算 roll: roll = arctan2(R[2,1], R[2,2])
        # roll = np.arctan2(current_rot[2,1], current_rot[2,2])
        # if np.abs(roll) > 1.0:  # 阈值 1.0 弧度（约57°），可根据需求调整
        #     print("np.abs(roll) > 1.0")
        #     return True

        current_rot = obs[3:12].reshape(3,3)
        roll = np.arctan2(current_rot[2,1], current_rot[2,2])
        pitch = np.arcsin(-current_rot[2,0])  # 另一种常见写法
        yaw = np.arctan2(current_rot[1,0], current_rot[0,0])

        # 设置阈值 (根据需求可改大/改小)
        roll_threshold = 5.0 * np.pi / 180.0   # 15度
        pitch_threshold = 5.0 * np.pi / 180.0  # 15度
        yaw_threshold = 5.0 * np.pi / 180.0    # 比如同样 15度
        att_ok = (abs(roll) < roll_threshold and
                abs(pitch) < pitch_threshold and
                abs(yaw) < yaw_threshold)
        
        # # 假设目标姿态的旋转矩阵（目标为单位矩阵，即无旋转）
        # desired_rot = self.desired_state[3:12].reshape(3, 3)

        # # 当前姿态旋转矩阵（从观测中获取）
        # current_rot = obs[3:12].reshape(3, 3)

        # # 计算相对旋转矩阵
        # R_diff = np.dot(desired_rot.T, current_rot)

        # # 计算角度差（注意数值可能超出 [-1,1]，因此需要 clip）
        # cos_theta = (np.trace(R_diff) - 1) / 2.0
        # cos_theta = np.clip(cos_theta, -1.0, 1.0)
        # rot_diff_angle = np.arccos(cos_theta)

        # # 设置阈值（例如：5度，转换为弧度）
        # orientation_threshold = 5.0 * np.pi / 180.0
        # print("rot_diff_angle: ", rot_diff_angle, " orientation_threshold: ", orientation_threshold)

        # # 判断是否小于阈值
        # att_ok = (rot_diff_angle < orientation_threshold)

        vel_error = np.linalg.norm(obs[12:15] - self.desired_state[12:15])
        print("roll: ", roll, ", pitch: ", pitch, ", self.roll_threshold: ", roll_threshold)
        if att_ok and pos_error < self.stable_pos_threshold/2 and vel_error < self.stable_vel_threshold/3:
            self.stable_count += 2
        elif att_ok and pos_error < self.stable_pos_threshold and vel_error < self.stable_vel_threshold:
            self.stable_count += 0.5
        else:
            self.stable_count = 0

        # 一旦满足 stable_steps_needed，则说明已经稳定悬停，done = True
        # if self.stable_count >= self.stable_steps_needed:
        #     print("Quadrotor hovered successfully!")
        #     return True

        
        # # 速度失控检查
        # if np.linalg.norm(obs[12:15]) > 8.0:  # 最大允许速度
        #     print("np.linalg.norm(obs[12:15]) > 8.0")
        #     return True
        
        return False
    
    def render(self, mode='human'):
        pass  # Gazebo 自带可视化

    def close(self):
        rospy.signal_shutdown("Environment closed")

    # def seed(self, seed=None):
    #     self._seed = seed
    #     np.random.seed(seed)
    #     random.seed(seed)
    #     return [seed]