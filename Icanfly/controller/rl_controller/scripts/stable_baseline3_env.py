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
            [0, 0, 3] + [1, 0, 0, 0, 1, 0, 0, 0, 1] + [0, 0, 0],
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
        state.pose.position.x = 0
        state.pose.position.y = 0
        state.pose.position.z = 0.2
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
            r_position +
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