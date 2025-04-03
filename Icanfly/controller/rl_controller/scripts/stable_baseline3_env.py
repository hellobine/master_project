
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces
# import rospy
# from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
# from nav_msgs.msg import Odometry
# from quadrotor_msgs.msg import ControlCommand, TrajectoryPoint, Trajectory
# from readerwriterlock import rwlock
# from gazebo_msgs.msg import ModelState
# from rotors_comm.msg import WindSpeed
# import threading
# from std_msgs.msg import Bool
# import random

# class QuadrotorEnv(gym.Env):
#     def __init__(self, namespace="drone"):
#         super(QuadrotorEnv, self).__init__()

#         # self.state_lock = rwlock.RWLockFair()
    
#         self.namespace = namespace  

#         self.mass = 0.68  # kg
#         self.gravity = 9.8066
#         self.min_thrust = 0 * self.mass * self.gravity
#         self.max_thrust =  4 * self.mass * self.gravity
#         self.max_angular_rate = 5.0

#         self.max_position_error = 4.0  # 米
#         self.max_velocity_error = 8.0  # 米

#         self.state_dim = 17  
#         self.action_dim = 4  # [推力, ωx, ωy, ωz]

        
#         self.observation_space = spaces.Box(
#             low=np.concatenate((
#                 np.full(3, -1.0),     # 位置差
#                 np.full(4, -1.0),     # 四元数 w,x,y,z          
#                 np.full(3, -1.0),     # 速度
#                 np.full(3, -1.0),     # angular_rate
#                 np.full(4, -1.0)      # 上一步动作
#             )),
#             high=np.concatenate((
#                 np.full(3, 1.0),
#                 np.full(4, 1.0),
#                 np.full(3, 1.0),
#                 np.full(3, 1.0),
#                 np.full(4, 1.0)
#             )),
#             dtype=np.float32
#         )

#         self.obs_real_low = np.concatenate((
#             np.full(3, -self.max_position_error),    # rel_pos
#             np.full(4, -1.0),                        # base_quat（假设已归一化）
#             np.full(3, -self.max_velocity_error),    # base_lin_vel
#             np.full(3, -self.max_angular_rate),      # base_ang_vel
#             np.full(1, self.min_thrust),             # last_actions
#             np.full(3, -self.max_angular_rate)
#         ))

#         self.obs_real_high = np.concatenate((
#             np.full(3, self.max_position_error),    # rel_pos
#             np.full(4, 1.0),                        # base_quat（假设已归一化）
#             np.full(3, self.max_velocity_error),    # base_lin_vel
#             np.full(3, self.max_angular_rate),      # base_ang_vel
#             np.full(1, self.max_thrust),             # last_actions
#             np.full(3, self.max_angular_rate),
#         ))

#         self.action_real_low = np.concatenate((
#             np.full(1, self.min_thrust),
#             np.full(3, -self.max_angular_rate)
#         ))

#         self.action_real_high = np.concatenate((
#             np.full(1, self.max_thrust),
#             np.full(3, self.max_angular_rate)
#         ))

#         self.action_space = spaces.Box(
#             low=np.array([-1]*self.action_dim, dtype=np.float32),
#             high=np.array([1]*self.action_dim, dtype=np.float32),
#             dtype=np.float32
#         )

#         if not rospy.core.is_initialized():
#             rospy.init_node(f'{self.namespace}_quadrotor_sb3_env', anonymous=True)
        

#         self.control_hz = 100.0

#         self.dt = 1/self.control_hz
#         self.rate = rospy.Rate(self.control_hz)
#         self.current_state = np.zeros(self.state_dim, dtype=np.float32)
#         self.desired_state = np.array(
#             [0, 1, 3] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0],
#             dtype=np.float32
#         )
   
#         self.max_episode_steps = 256
#         self.step_count = 0
#         self.episode_reward = 0

#         self.prev_real_thrust = None
#         self.prev_real_bodyrates = None

#         self.prev_action_ = [0, 0, 0, 0]
#         self.prev_rel_pos = None  # 上一步的相对位置

#         # 奖励缩放系数与偏航参数（各项均已乘上时间步长 dt，可按实际需求调整）
#         self.s_target = 10.0
#         self.s_smooth = 0.0001
#         self.s_yaw = 0.01
#         self.s_angular = -0.0002
#         self.s_crash = -10
#         self.yaw_lambda = -10


#         self.prev_pos_error = None
#         self.observation_space_norm = np.array(
#             [0, 0, 4] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0] + [0, 0, 0, 0],
#             dtype=np.float32
#         )
   

#         self.angular_x = 0
#         self.angular_y = 0
#         self.angular_z = 0

#         self.stable_count = 0
#         self.stable_steps_needed = 5   # 连续满足条件多少步后判定成功
#         self.stable_pos_threshold = 0.2  # 位置误差小于该值
#         self.stable_vel_threshold = 0.5  # 速度误差小于该值
#         self.reference_trajectory_ = None  # 使用列表模拟队列


#         # self.pnh_ = rospy.get_namespace()  # 这里用 ROS 的命名空间代替参数句柄
  
#         self.reference_state_ = None  # 例如一个包含 position 属性的对象，position 为 numpy 数组
#         self.kPositionJumpTolerance_ = 0.1  # 示例值，单位与 position 相同
#         self.first_time_in_new_state_=True

    
#         self.odom_sub = rospy.Subscriber(f'/{self.namespace}/ground_truth/odometry', Odometry, self.odom_callback)
#         self.cmd_pub = rospy.Publisher(f'/{self.namespace}/control_command', ControlCommand, queue_size=3)
#         self.arm_pub = rospy.Publisher(f'/{self.namespace}/bridge/arm', Bool, queue_size=1)
#         self.reference_state_sub = rospy.Subscriber('/autopilot/trajectory', Trajectory, self.reference_trajectory_callback)
#         self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)

#         # rospy.sleep(1.0)
    
#         if not rospy.is_shutdown():
#             msg = Bool(data=True)
#             self.arm_pub.publish(msg)
#             rospy.loginfo("Published arm message: true")

#         self._reset_drone_pose()
#         rospy.sleep(0.3)

#         # print("reset@@@@@@@@@@@@@")
    
#     def _quaternion_to_rotation_matrix(self, x, y, z, w):
#         """将四元数转换为旋转矩阵（3x3），注意四元数顺序为 (x, y, z, w)"""
#         R = np.array([
#             [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
#             [2*(x*y + z*w),           1 - 2*(x**2 + z**2),    2*(y*z - x*w)],
#             [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
#         ], dtype=np.float32)
#         return R
    
#     def _normalize_obs(self, obs):
#         """
#         将原始观测值按预设尺度线性映射到 [-1, 1]。
#         公式： norm = 2 * (obs - low) / (high - low) - 1
#         """
#         norm_obs = (obs - self.obs_real_low) / (self.obs_real_high - self.obs_real_low + 1e-8)
#         norm_obs = norm_obs * 2 - 1
#         return norm_obs.astype(np.float32)
        
#     def odom_callback(self, msg):
#         # with self.state_lock.gen_wlock():
#             pos = [msg.pose.pose.position.x, 
#                    msg.pose.pose.position.y,
#                    msg.pose.pose.position.z]

#             qx = msg.pose.pose.orientation.x
#             qy = msg.pose.pose.orientation.y
#             qz = msg.pose.pose.orientation.z
#             qw = msg.pose.pose.orientation.w
#             rot_mat = [qw,qx,qy,qz]

#             lin_vel = [msg.twist.twist.linear.x,
#                        msg.twist.twist.linear.y,
#                        msg.twist.twist.linear.z]
            
#             ang_vel = [msg.twist.twist.angular.x,
#                        msg.twist.twist.angular.y,
#                        msg.twist.twist.angular.z]

#             self.current_state = np.array(
#                 pos + rot_mat + lin_vel + ang_vel, dtype=np.float32
#             )

#     def reference_trajectory_callback(self, msg):
#         pass
  

#     def step(self, action):
#         self.step_count += 1
#         self._publish_action(action)
#         self.rate.sleep()
#         # with self.state_lock.gen_rlock():
#         current_state = self.current_state.copy()

#         self.observation_space_norm, reward = self._compute_reward(current_state, action)
    
#         check_done = self._check_done(current_state)

#         self.episode_reward += reward

#         if self.step_count >= self.max_episode_steps:
#             if self.step_count >= self.max_episode_steps:
#                 terminated = False  # 非任务失败
#                 truncated = True   # 由于达到最大步数
#             else:
#                 terminated = True  # 任务失败，如超出范围
#                 truncated = False
 
#             # 在所有结束条件下，都返回 episode 信息
#             info = {"reward": reward, "episode": {"r": self.episode_reward, "l": self.step_count}}

#         else:
#             terminated = False
#             truncated = False
#             info = {"reward": reward}

#         # if terminated or truncated:
#         #     self.episode_reward = 0
#         #     self.step_count = 0

#         return self.observation_space_norm, reward, terminated, truncated, info

#     def reset(self, **kwargs):
#         self.step_count = 0
#         self.episode_reward = 0
#         self.prev_action_ = [0,0,0,0]

#         self._reset_drone_pose()
        

#         curr_rel_pos = self.current_state[0:3] - self.desired_state[0:3]
#         base_quat = self.current_state[3:7]
#         base_lin_vel = self.current_state[7:10]
#         base_ang_vel = self.current_state[10:13]

#         obs = np.concatenate((
#             curr_rel_pos,
#             base_quat,
#             base_lin_vel,
#             base_ang_vel,
#             np.array(self.prev_action_)
#         ))

#         norm_obs = self._normalize_obs(obs)
#         return norm_obs, {}
    

#     def _reset_drone_pose(self):
#         init_position = np.array([
#             np.random.uniform(0.0, 0.0),  # x 范围 [-1, 1] m
#             np.random.uniform(0.0, 0.0),  # y 范围 [-1, 1] m
#             np.random.uniform(2, 4)    # z 范围 [0.5, 1.5] m，避免过高或过低
#         ])

#         init_orientation = np.random.uniform(-3, 3, size=3)  
#         init_angular_velocity = np.random.uniform(-1, 1, size=3)  

#         state = ModelState()
#         state.model_name = self.namespace  # 确保命名空间匹配 Gazebo 的无人机模型
#         state.pose.position.x = 0
#         state.pose.position.y = 0
#         state.pose.position.z = 2
#         state.pose.orientation.x = 0
#         state.pose.orientation.y = 0
#         state.pose.orientation.z = 0
#         state.pose.orientation.w = init_orientation[2]  # 保持单位四元数

#         # state.twist.linear.x=init_velocity[0]
#         # state.twist.linear.y=init_velocity[1]
#         # state.twist.linear.z=init_velocity[2]

#         # state.twist.angular.x=init_angular_velocity[0]
#         # state.twist.angular.y=init_angular_velocity[1]
#         # state.twist.angular.z=init_angular_velocity[2]

#         pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
#         pub.publish(state)

#         # rospy.sleep(0.01)

#     def _publish_action(self, action):
#         cmd = ControlCommand()
#         cmd.armed = True
#         cmd.control_mode = ControlCommand.BODY_RATES
#         # cmd.collective_thrust = float(np.clip(action[0], self.min_thrust, self.max_thrust))
#         # cmd.collective_thrust = action[0]

#         thrust = ((action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
#         cmd.collective_thrust = thrust

#         # 角速度归一化映射到 [-max_angular_rate, max_angular_rate]
#         bodyrates = np.clip(action[1:], -1, 1) * self.max_angular_rate

#         cmd.bodyrates.x = bodyrates[0]
#         cmd.bodyrates.y = bodyrates[1]
#         cmd.bodyrates.z = bodyrates[2]
#         # print("cmd: ", cmd)
#         self.cmd_pub.publish(cmd)

#     def _compute_reward1(self, current_state, curr_action):
#         curr_pos_error = current_state[0:3]  - self.desired_state[0:3]
#         curr_ori_error = current_state[3:7]  - self.desired_state[3:7]
#         curr_vel_error = current_state[7:10] - self.desired_state[7:10]

#         observation_space_ = np.array(list(curr_pos_error) + list(curr_vel_error) + list(curr_ori_error) + list(self.prev_action_), dtype=np.float32)
        
#         curr_pos_error_norm = np.linalg.norm(curr_pos_error)
#         curr_ori_error_norm = np.linalg.norm(curr_ori_error)
#         curr_vel_error_norm = np.linalg.norm(curr_vel_error)
#         action_diff_norm = np.linalg.norm(curr_action - self.prev_action_)

#         if self.prev_pos_error is not None:
#             progress_pos_error = self.prev_pos_error - curr_pos_error_norm
#         else:
#             progress_pos_error = 0
            
#         r_position = np.exp(-0.3 * curr_pos_error_norm)
#         r_attitude = np.exp(-0.3 * curr_ori_error_norm)
#         r_velocity = np.exp(-0.3 * curr_vel_error_norm)
#         r_smooth   = np.exp(-0.3 * action_diff_norm)
#         r_progress = np.tanh(0.3 * progress_pos_error)

#         r_progress_weight = 1 
#         r_position_weight = 4
#         r_velocity_weight = 1
#         r_attitude_weight = 0.0
#         r_smooth_weight=0.4

#         # print("reward: ", r_progress_weight*r_progress, r_position_weight*r_position, r_velocity_weight*r_velocity, r_attitude_weight*r_attitude, r_smooth_weight*r_smooth)
        
#         total_reward = (
#             r_progress_weight*r_progress +
#             r_position_weight*r_position +
#             r_velocity_weight*r_velocity +
#             r_attitude_weight*r_attitude +
#             r_smooth_weight*r_smooth 
#         )

        

#         self.prev_action_ = curr_action.copy()
#         self.prev_pos_error = curr_pos_error_norm

#         norm_obs = self._normalize_obs(observation_space_)
#         # print("observation_space_: ", observation_space_)
#         # print("norm_obs: ", norm_obs)

#         return norm_obs, total_reward


#     def _compute_reward(self, current_state, curr_action):
#         # 1. 当前相对位置误差：当前位置与目标位置之差
#         curr_rel_pos = current_state[0:3] - self.desired_state[0:3]

#         # 2. 接近目标奖励
#         if self.prev_rel_pos is not None:
#             r_target = np.linalg.norm(self.prev_rel_pos)**2 - np.linalg.norm(curr_rel_pos)**2
#         else:
#             r_target = 0.0

#         # 3. 动作平滑奖励：当前动作与上一次动作的平方差之和
#         r_smooth = np.sum((curr_action - np.array(self.prev_action_))**2)

#         # 4. 偏航角奖励：从四元数（顺序 [qw, qx, qy, qz]）中计算 yaw
#         qw, qx, qy, qz = current_state[3:7]
#         yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
#         theta = yaw * 180 / np.pi  # 转换为角度
#         if theta > 180:
#             phi = theta - 360
#         else:
#             phi = theta
#         tilde_theta = phi * np.pi / 180  # 转换为弧度
#         r_yaw = np.exp(self.yaw_lambda * abs(tilde_theta))

#         # 5. 角速度奖励：假定角速度存储在 current_state 的索引 10:13
#         omega = current_state[10:13]
#         r_angular = np.linalg.norm(omega / np.pi)

#         # 6. 撞机奖励：若满足撞机条件，则为 1，否则为 0
#         r_crash = 1.0 if self._check_done(current_state) else 0.0

#         # 7. 总奖励：各项奖励按预设缩放系数加权求和
#         total_reward = self.dt *(
#             self.s_target * r_target +
#             self.s_smooth * r_smooth +
#             self.s_yaw * r_yaw +
#             self.s_angular * r_angular +
#             self.s_crash * r_crash
#         )

#         base_quat = current_state[3:7]
#         base_lin_vel = current_state[7:10]
#         base_ang_vel = current_state[10:13]
        

#         # 构造新的观测：包含当前相对位置、线速度（索引 7:10）、四元数（索引 3:7）及上一步动作
#         obs = np.concatenate((
#             curr_rel_pos,
#             base_quat,
#             base_lin_vel,
#             base_ang_vel,
#             np.array(self.prev_action_)
#         ))

#         norm_obs = self._normalize_obs(obs)

#         # 更新历史记录
#         self.prev_action_ = curr_action.copy()
#         self.prev_rel_pos = curr_rel_pos.copy()

#         return norm_obs, total_reward

#     def _check_done(self, curr_state):

#         # x, y, z = curr_state[0], curr_state[1], curr_state[2]
#         # if x < -1 or x > 1 or y < -1 or y > 1 or z < 0 or z > 3:
#         #     return True
        
#         pos_error = np.linalg.norm(curr_state[0:3] - self.desired_state[0:3])
#         if pos_error > self.max_position_error:
#             return True
        
#         qw = curr_state[3]
#         qx = curr_state[4]
#         qy = curr_state[5]
#         qz = curr_state[6]
#         roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
#         # if np.abs(roll) > 2.8:
#         #     return True

#         if np.linalg.norm(curr_state[7:10]) > self.max_velocity_error: 
#             print("vel > ", self.max_velocity_error)
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
from readerwriterlock import rwlock
from gazebo_msgs.msg import ModelState
from rotors_comm.msg import WindSpeed
import threading
from std_msgs.msg import Bool
import random

class QuadrotorEnv(gym.Env):
    def __init__(self, namespace="drone"):
        super(QuadrotorEnv, self).__init__()

        self.state_lock = rwlock.RWLockFair()
    
        self.namespace = namespace  

        self.mass = 0.73  # kg
        self.gravity = 9.8066
        self.min_thrust = 0 * self.mass * self.gravity
        self.max_thrust =  4 * self.mass * self.gravity
        self.max_angular_rate = 5.0

        self.max_position_error = 5.0  # 米
        self.max_velocity_error = 8.0  # 米

        self.state_dim = 17  
        self.action_dim = 4  # [推力, ωx, ωy, ωz]

        
        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.full(3, -1.0),     # 位置差
                np.full(4, -1.0),     # 四元数 w,x,y,z          
                np.full(3, -1.0),     # 速度
                np.full(3, -1.0),     # angular_rate
                np.full(1, -1.0),
                np.full(3, -1.0)      # 上一步动作
            )),
            high=np.concatenate((
                np.full(3, 1.0),
                np.full(4, 1.0),
                np.full(3, 1.0),
                np.full(3, 1.0),
                np.full(1, 1.0),
                np.full(3, 1.0)      # 上一步动作
            )),
            dtype=np.float32
        )

        self.obs_real_low = np.concatenate((
            np.full(3, -self.max_position_error),    # rel_pos
            np.full(4, -1.0),                        # base_quat（假设已归一化）
            np.full(3, -self.max_velocity_error),    # base_lin_vel
            np.full(3, -self.max_angular_rate),      # base_ang_vel
            np.full(1, self.min_thrust),             # last_actions
            np.full(3, -self.max_angular_rate)
        ))

        self.obs_real_high = np.concatenate((
            np.full(3, self.max_position_error),    # rel_pos
            np.full(4, 1.0),                        # base_quat（假设已归一化）
            np.full(3, self.max_velocity_error),    # base_lin_vel
            np.full(3, self.max_angular_rate),      # base_ang_vel
            np.full(1, self.max_thrust),             # last_actions
            np.full(3, self.max_angular_rate),
        ))

        self.action_real_low = np.concatenate((
            np.full(1, self.min_thrust),
            np.full(3, -self.max_angular_rate)
        ))

        self.action_real_high = np.concatenate((
            np.full(1, self.max_thrust),
            np.full(3, self.max_angular_rate)
        ))

        self.action_space = spaces.Box(
            low=np.array([-1]*self.action_dim, dtype=np.float32),
            high=np.array([1]*self.action_dim, dtype=np.float32),
            dtype=np.float32
        )

        if not rospy.core.is_initialized():
            rospy.init_node(f'{self.namespace}_quadrotor_sb3_env', anonymous=True)
        

        self.control_hz = 100.0

        self.dt = 1/self.control_hz
        self.rate = rospy.Rate(self.control_hz)
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        # target_pos_=[random.randint[-1,1], random.randint[-1,1], random.randint[1,4]]
        self.desired_state = np.array(
            [1, 1, 3] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0],
            dtype=np.float32
        )
   
        self.max_episode_steps = 64
        self.step_count = 0
        self.episode_reward = 0

        self.prev_real_thrust = None
        self.prev_real_bodyrates = None

        self.prev_action_ = [0, 0, 0, 0]
        self.prev_rel_pos = None  # 上一步的相对位置

        # 奖励缩放系数与偏航参数（各项均已乘上时间步长 dt，可按实际需求调整）
        self.s_target = 2
        self.s_smooth = 0.0001
        self.s_yaw = 0.01
        self.s_angular = -0.0002
        self.s_crash = -10
        self.yaw_lambda = 0.3


        self.prev_pos_error = None
        self.observation_space_norm = np.array(
            [0, 0, 2] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0] + [0, 0, 0, 0],
            dtype=np.float32
        )

        self.prev_position = [0,0,0]
   

        self.angular_x = 0
        self.angular_y = 0
        self.angular_z = 0

        self.stable_count = 0
        self.stable_steps_needed = 5   # 连续满足条件多少步后判定成功
        self.stable_pos_threshold = 0.2  # 位置误差小于该值
        self.stable_vel_threshold = 0.5  # 速度误差小于该值
        self.reference_trajectory_ = None  # 使用列表模拟队列


        # self.pnh_ = rospy.get_namespace()  # 这里用 ROS 的命名空间代替参数句柄
  
        self.reference_state_ = None  # 例如一个包含 position 属性的对象，position 为 numpy 数组
        self.kPositionJumpTolerance_ = 0.1  # 示例值，单位与 position 相同
        self.first_time_in_new_state_=True

    
        self.odom_sub = rospy.Subscriber(f'/{self.namespace}/ground_truth/odometry', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher(f'/{self.namespace}/control_command', ControlCommand, queue_size=3)
        self.arm_pub = rospy.Publisher(f'/{self.namespace}/bridge/arm', Bool, queue_size=1)
        self.reference_state_sub = rospy.Subscriber('/autopilot/trajectory', Trajectory, self.reference_trajectory_callback)
        self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)

        rospy.sleep(1)# can cause agent can control, because of aremd start failed
    
        if not rospy.is_shutdown():
            msg = Bool(data=True)
            self.arm_pub.publish(msg)
            rospy.loginfo("Published arm message: true")

        self._reset_drone_pose()

        # print("reset@@@@@@@@@@@@@")
    
    def _quaternion_to_rotation_matrix(self, x, y, z, w):
        """将四元数转换为旋转矩阵（3x3），注意四元数顺序为 (x, y, z, w)"""
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2),    2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        return R
    
    def _normalize_obs(self, obs):
        """
        将原始观测值按预设尺度线性映射到 [-1, 1]。
        公式： norm = 2 * (obs - low) / (high - low) - 1
        """
        # print("原始 obs: ", obs)
        
        # 归一化前13维
        norm_obs_first = (obs[0:13] - self.obs_real_low[0:13]) / (self.obs_real_high[0:13] - self.obs_real_low[0:13] + 1e-8)
        norm_obs_first = norm_obs_first * 2 - 1
        # print("归一化后的前13维: ", norm_obs_first)
        
        # 后面的[13:17]部分直接使用，因为预设动作输出已处于[-1,1]
        norm_obs_second = obs[13:17]
        # print("直接使用的后4维 (动作部分): ", norm_obs_second)
        
        # 拼接两个部分
        norm_obs = np.concatenate((norm_obs_first, norm_obs_second))
        # print("最终归一化后的 obs: ", norm_obs)
        
        return norm_obs.astype(np.float32)
        
    def odom_callback(self, msg):
        with self.state_lock.gen_wlock():
            pos = [msg.pose.pose.position.x, 
                   msg.pose.pose.position.y,
                   msg.pose.pose.position.z]

            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            rot_mat = [qw,qx,qy,qz]

            lin_vel = [msg.twist.twist.linear.x,
                       msg.twist.twist.linear.y,
                       msg.twist.twist.linear.z]
            
            ang_vel = [msg.twist.twist.angular.x,
                       msg.twist.twist.angular.y,
                       msg.twist.twist.angular.z]

            self.current_state = np.array(
                pos + rot_mat + lin_vel + ang_vel, dtype=np.float32
            )

    def reference_trajectory_callback(self, msg):
        pass
  

    def step(self, action):
        self.step_count += 1
        self._publish_action(action)
        self.rate.sleep()

        with self.state_lock.gen_rlock():
            current_state = self.current_state.copy()

        self.observation_space_norm, reward = self._compute_reward(current_state, action)
    
        check_done = self._check_done(current_state)

        self.episode_reward += reward

        if self.step_count >= self.max_episode_steps or check_done:
            if self.step_count >= self.max_episode_steps:
                terminated = False  # 非任务失败
                truncated = True   # 由于达到最大步数
            else:
                terminated = True  # 任务失败，如超出范围
                truncated = False
 
            # 在所有结束条件下，都返回 episode 信息
            info = {"reward": reward, "episode": {"r": self.episode_reward, "l": self.step_count}}

        else:
            terminated = False
            truncated = False
            info = {"reward": reward}

        # if terminated or truncated:
        #     self.episode_reward = 0
        #     self.step_count = 0

        return self.observation_space_norm, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.episode_reward = 0
        self.prev_action_ = [0,0,0,0]
 

        self._reset_drone_pose()
        
        self.prev_position = self.current_state[0:3]
        
        curr_rel_pos = self.current_state[0:3] - self.desired_state[0:3]
        base_quat = self.current_state[3:7]
        base_lin_vel = self.current_state[7:10]
        base_ang_vel = self.current_state[10:13]

        obs = np.concatenate((
            curr_rel_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            np.array(self.prev_action_)
        ))

        norm_obs = self._normalize_obs(obs)
        return norm_obs, {}
    

    def _reset_drone_pose(self):
        init_position = np.array([
            np.random.uniform(-1, 1),  # x 范围 [-1, 1] m
            np.random.uniform(-1, 1),  # y 范围 [-1, 1] m
            np.random.uniform(2, 4)    # z 范围 [0.5, 1.5] m，避免过高或过低
        ])

        init_orientation = np.random.uniform(-3, 3, size=3)  
        init_angular_velocity = np.random.uniform(-1, 1, size=3)  

        state = ModelState()
        state.model_name = self.namespace  # 确保命名空间匹配 Gazebo 的无人机模型
        state.pose.position.x = init_position[0]
        state.pose.position.y = init_position[1]
        state.pose.position.z = 1
        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = 0
        state.pose.orientation.w = init_orientation[2]  # 保持单位四元数

        # state.twist.linear.x=init_velocity[0]
        # state.twist.linear.y=init_velocity[1]
        # state.twist.linear.z=init_velocity[2]

        # state.twist.angular.x=init_angular_velocity[0]
        # state.twist.angular.y=init_angular_velocity[1]
        # state.twist.angular.z=init_angular_velocity[2]

        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        pub.publish(state)

        rospy.sleep(0.1)

    def _publish_action(self, action):
        cmd = ControlCommand()
        cmd.armed = True
        cmd.control_mode = ControlCommand.BODY_RATES
        # cmd.collective_thrust = float(np.clip(action[0], self.min_thrust, self.max_thrust))
        # cmd.collective_thrust = action[0]

        thrust = ((action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        cmd.collective_thrust = thrust

        # 角速度归一化映射到 [-max_angular_rate, max_angular_rate]
        bodyrates = np.clip(action[1:], -1, 1) * self.max_angular_rate

        cmd.bodyrates.x = bodyrates[0]
        cmd.bodyrates.y = bodyrates[1]
        cmd.bodyrates.z = bodyrates[2]
        # print("cmd: ", cmd)
        self.cmd_pub.publish(cmd)

    def _compute_reward1(self, current_state, curr_action):
        curr_pos_error = current_state[0:3]  - self.desired_state[0:3]
        curr_ori_error = current_state[3:7]  - self.desired_state[3:7]
        curr_vel_error = current_state[7:10] - self.desired_state[7:10]

        observation_space_ = np.array(list(curr_pos_error) + list(curr_vel_error) + list(curr_ori_error) + list(self.prev_action_), dtype=np.float32)
        
        curr_pos_error_norm = np.linalg.norm(curr_pos_error)
        curr_ori_error_norm = np.linalg.norm(curr_ori_error)
        curr_vel_error_norm = np.linalg.norm(curr_vel_error)
        action_diff_norm = np.linalg.norm(curr_action - self.prev_action_)

        if self.prev_pos_error is not None:
            progress_pos_error = self.prev_pos_error - curr_pos_error_norm
        else:
            progress_pos_error = 0
            
        r_position = np.exp(-0.3 * curr_pos_error_norm)
        r_attitude = np.exp(-0.3 * curr_ori_error_norm)
        r_velocity = np.exp(-0.3 * curr_vel_error_norm)
        r_smooth   = np.exp(-0.3 * action_diff_norm)
        r_progress = np.tanh(0.3 * progress_pos_error)

        r_progress_weight = 1 
        r_position_weight = 4
        r_velocity_weight = 1
        r_attitude_weight = 0.0
        r_smooth_weight=0.4

        # print("reward: ", r_progress_weight*r_progress, r_position_weight*r_position, r_velocity_weight*r_velocity, r_attitude_weight*r_attitude, r_smooth_weight*r_smooth)
        
        total_reward = (
            r_progress_weight*r_progress +
            r_position_weight*r_position +
            r_velocity_weight*r_velocity +
            r_attitude_weight*r_attitude +
            r_smooth_weight*r_smooth 
        )

        

        self.prev_action_ = curr_action.copy()
        self.prev_pos_error = curr_pos_error_norm

        norm_obs = self._normalize_obs(observation_space_)

        return norm_obs, total_reward


    def _compute_reward(self, current_state, curr_action):
        # 1. 当前相对位置误差：当前位置与目标位置之差
        
        curr_pos = current_state[0:3]

        prev_dist = np.linalg.norm(self.desired_state[0:3] - self.prev_position)
        curr_dist = np.linalg.norm(self.desired_state[0:3] - curr_pos)
        r_progress_dis = prev_dist - curr_dist

        r_target = 2*np.tanh(0.5*r_progress_dis)

        if r_target>0.6:
            print("curr_pos: ", curr_pos)
            print("self.prev_position: ", self.prev_position)
            print("prev_dist: ", prev_dist)
            print("curr_dist: ", curr_dist)
            print("r_target: ", r_target)
    

        curr_rel_pos = current_state[0:3] - self.desired_state[0:3] 
        curr_rel_dis = np.linalg.norm(current_state[0:3] - self.desired_state[0:3])
        r_position = np.exp(-0.3 * curr_rel_dis)

        # 2. 接近目标奖励
        # if self.prev_rel_pos is not None:
            # r_target = np.linalg.norm(self.prev_rel_pos)**2 - np.linalg.norm(curr_rel_pos)**2
        # else:
            # r_target = 0.0

        # 3. 动作平滑奖励：当前动作与上一次动作的平方差之和
        r_smooth = np.sum((curr_action - np.array(self.prev_action_))**2)

        # 4. 偏航角奖励：从四元数（顺序 [qw, qx, qy, qz]）中计算 yaw
        qw, qx, qy, qz = current_state[3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        theta = yaw * 180 / np.pi  # 转换为角度
        if theta > 180:
            phi = theta - 360
        else:
            phi = theta
        tilde_theta = phi * np.pi / 180  # 转换为弧度
        r_yaw = np.exp(self.yaw_lambda * abs(tilde_theta))

        # 5. 角速度奖励：假定角速度存储在 current_state 的索引 10:13
        omega = current_state[10:13]
        r_angular = np.linalg.norm(omega / np.pi)

        # 6. 撞机奖励：若满足撞机条件，则为 1，否则为 0
        r_crash = 1.0 if self._check_done(current_state) else 0.0
        # print("self._check_done(current_state): ", self._check_done(current_state))
        # print("r_crash: ", r_crash)

        # 7. 总奖励：各项奖励按预设缩放系数加权求和
        # print(r_position)
        # print("self.s_target * r_target: ", self.s_target * r_target)
        total_reward = (
            1 * r_position +
            self.s_target  * r_target +
            self.s_smooth * r_smooth +
            self.s_yaw * r_yaw +
            self.s_angular * r_angular +
            self.s_crash * r_crash
        )
        print("1 * r_position: ", 1 * r_position, "self.s_target * r_target: ", self.s_target * r_target, "self.s_crash * r_crash: ", self.s_crash * r_crash)
        # print("[7] 各项加权值：")
        # print(f"    r_target weighted: {self.s_target * r_target}")
        # print(f"    r_smooth weighted: {self.s_smooth * r_smooth}")
        # print(f"    r_yaw weighted: {self.s_yaw * r_yaw}")
        # print(f"    r_angular weighted: {self.s_angular * r_angular}")
        # print(f"    r_crash weighted: {self.s_crash * r_crash}")
        # print(f"[7] total_reward = {total_reward}")


        base_quat = current_state[3:7]
        base_lin_vel = current_state[7:10]
        base_ang_vel = current_state[10:13]
        

        obs = np.concatenate((
            curr_rel_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            np.array(self.prev_action_)
        ))


        norm_obs = self._normalize_obs(obs)

        # 更新历史记录
        self.prev_action_ = curr_action.copy()
        self.prev_rel_pos = curr_rel_pos.copy()
        self.prev_position = curr_pos.copy()

        return norm_obs, total_reward



    def _check_done(self, curr_state):

        # x, y, z = curr_state[0], curr_state[1], curr_state[2]
        # if x < -1 or x > 1 or y < -1 or y > 1 or z < 0 or z > 3:
        #     return True
        
        pos_error = np.linalg.norm(curr_state[0:3] - self.desired_state[0:3])
        if pos_error > self.max_position_error:
            return True
        
        qw = curr_state[3]
        qx = curr_state[4]
        qy = curr_state[5]
        qz = curr_state[6]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        if np.abs(roll) > 1.8:
            return True

        if np.linalg.norm(curr_state[7:10]) > self.max_velocity_error: 
            print("vel > ", self.max_velocity_error)
            return True
        
        return False
    
    def render(self, mode='human'):
        pass  # Gazebo 自带可视化

    def close(self):
        rospy.signal_shutdown("Environment closed")

