#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import numpy as np
from gym import spaces
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState

class QuadrotorEnv(gym.Env):
    def __init__(self):
        """
        继承自 gym.Env，定义强化学习环境，模拟四旋翼无人机的控制交互。
        """
        super(QuadrotorEnv, self).__init__()

        # --------------- 1. 定义状态与动作空间 ---------------
        self.state_dim = 10   # [p_x, p_y, p_z, q_x, q_y, q_z, q_w, v_x, v_y, v_z]
        self.action_dim = 4   # [T, ω_x, ω_y, ω_z]

        obs_high = np.array([np.inf] * self.state_dim, dtype=np.float32)
        act_high = np.array([1.0] * self.action_dim, dtype=np.float32)

        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        # --------------- 2. ROS 相关初始化 ---------------
        # rospy.init_node('quadrotor_rl_env', anonymous=True)

        # 订阅无人机状态（位置 & 速度）
        self.odom_sub = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odom_callback)

        # 发布控制指令
        self.cmd_pub = rospy.Publisher('/hummingbird/control_command', ControlCommand, queue_size=1)
# /hummingbird/autopilot/trajectory
        # 控制频率
        self.control_hz = 100.0
        self.rate = rospy.Rate(self.control_hz)

        # --------------- 3. 初始化内部状态 ---------------
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.desired_state = np.array([0, 0, 3, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self.episode_time = 0.0
        self.max_episode_time = 5.0   # 限制每个 episode 最长 5 秒


        self.max_position_error = 5.0  # 例如，允许偏差超过 5 米则重置

        self.max_position_error = 5.0  # 位置偏差阈值（米）
        self.flip_threshold = 1.0      # 侧翻判断阈值，修改为 1.0 rad（约57°）
        self.min_velocity = 0.1        # 低速阈值（m/s）
        self.flip_time_limit = 10     # 持续侧翻+低速状态的时间限制（秒）
        self.flip_timer = 0.0          # 累计时间


    def odom_callback(self, msg):
        """
        解析无人机的里程计信息（Odometry）。
        """
        p_x, p_y, p_z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
        q_x, q_y, q_z, q_w = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        v_x, v_y, v_z = msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z

        # print("p_x   -> "+str(p_x))

        self.current_state = np.array([p_x, p_y, p_z, q_x, q_y, q_z, q_w, v_x, v_y, v_z], dtype=np.float32)

    def step(self, action):
        """
        执行动作，并返回 (obs, reward, done, info)。
        """
        self._publish_action(action)
        self.rate.sleep()  # 保持 50Hz 运行
        self.episode_time += 1.0 / self.control_hz

        obs = self._get_obs()
        reward = self._compute_reward(obs)

        # 计算当前无人机与目标点的距离（只比较位置部分）
        pos_error = np.linalg.norm(self.current_state[0:3] - self.desired_state[0:3])
        # 如果距离超过阈值，则将 done 标记为 True，并给予额外惩罚
        if pos_error > self.max_position_error:
            reward -= 100.0  # 惩罚项，可根据需要调整
            done = True
            print(f"Reset triggered due to large position error: {pos_error:.2f} m")
        elif pos_error < 0.2:
            reward += 1000.0  # 额外奖励，根据需要调整
            done = True
            print("Target reached: episode terminated successfully.")
        else:
            done = self.episode_time >= self.max_episode_time

        return obs, reward, done, {}

    def reset(self):
        """
        重置环境，等待系统恢复初始状态。
        """
        
        try:
            self._reset_drone_pose()  # 重置控制器状态
            print("World reset successfully!")
            # 建议增加短暂延时确保物理引擎稳定
            rospy.sleep(1.5)  
        except rospy.ServiceException as e:
            print(f"World reset failed: {e}")

        self.episode_time = 0.0
        return self._get_obs()
    

    def _check_flip(self, dt):
        """
        判断无人机是否侧翻且长时间停滞：
          - 将四元数转换为 Euler 角，判断 roll、pitch 是否超过阈值
          - 判断线速度是否低于阈值
        dt 为当前 step 的时间间隔（秒）
        """
       
        # 判断当前速度是否很低
        speed = np.linalg.norm(self.current_state[7:10])
        low_speed = (speed < self.min_velocity)
        if low_speed:
            self.flip_timer += dt
        else:
            self.flip_timer = 0.0
        return self.flip_timer >= self.flip_time_limit
    
    def _reset_drone_pose(self):
        """将无人机精确复位到初始位置"""
        pose_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        state = ModelState()
        state.model_name = "hummingbird"
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 0.1
        state.pose.orientation.w = 1.0
        pose_pub.publish(state)


    def _publish_action(self, action):
        """
        将动作 [T, ω_x, ω_y, ω_z] 发送至 ROS 控制话题。
        """
        command = ControlCommand()
        command.armed = True
        # 这里的 control_mode 根据你的系统定义进行设置，比如 BODY_RATES
        command.control_mode = ControlCommand.BODY_RATES  
        command.collective_thrust = action[0]   # 推力
        # 将角速度拆分到 bodyrates 的各个分量
        command.bodyrates.x = action[1]
        command.bodyrates.y = action[2]
        command.bodyrates.z = action[3]
        print("collective_thrust   -> "+str(command.collective_thrust))
        # 发布正确类型的控制命令
        self.cmd_pub.publish(command)

    def _get_obs(self):
        """
        获取当前观测状态，可以是状态误差或原始状态。
        """
        return self.current_state

    # def _compute_reward(self, obs):
    #     """
    #     计算奖励，包括位置误差、速度误差、姿态误差等。
    #     """
    #     pos_error = np.linalg.norm(obs[0:3])  # 位置误差
    #     quat_error = 2 * np.arccos(np.clip(np.abs(obs[3:7].dot(self.desired_state[3:7])), -1.0, 1.0))  # 四元数误差
    #     vel_error = np.linalg.norm(obs[7:10])  # 速度误差

    #     # 调整权重，尝试使位置误差更敏感
    #     error = 3 * pos_error + 0.5 * quat_error + 0.1 * vel_error

    #     # 使用反比函数作为奖励信号
    #     reward = 1.0 / (error + 1.0)

    #     return reward

    def _compute_reward(self, obs):
        """
        奖励函数示例：
        - 对位置和速度的平方误差进行惩罚
        - 当 UAV 靠近目标时（误差小于阈值），给予正奖励
        """
        # 从观察值中提取当前位置和速度
        pos = obs[0:3]
        vel = obs[7:10]
        
        # 目标位置，例如 [0, 0, 3]
        target_pos = self.desired_state[0:3]
        pos_error = np.linalg.norm(pos - target_pos)
        
        # 期望速度为零（例如在悬停或稳定着陆时）
        vel_error = np.linalg.norm(vel)
        
        # 权重参数（这些参数需要根据具体任务进行调节）
        alpha = 3.0  # 位置误差权重
        beta = 5.0   # 速度误差权重
        # 若有控制输入成本，可以设置 gamma，对应 u 的平方（这里没有使用）
        # gamma = 0.01
        
        # 基础奖励为负的误差平方和
        reward = - (alpha * pos_error**2 + beta * vel_error**2)
        
        # 如果位置误差足够小（例如小于 0.2 米），则给予额外奖励
        if pos_error < 0.6 and (target_pos[2] - pos[2]) > 0.1:
            reward += 1000.0

        return reward

    # def _compute_reward(self, obs):
    #     # 提取位置和速度
    #     pos = obs[0:3]
    #     vel = obs[7:10]
        
    #     # 目标位置
    #     target_pos = self.desired_state[0:3]
    #     pos_error = np.linalg.norm(pos - target_pos)
    #     vel_error = np.linalg.norm(vel)
        
    #     # 权重参数
    #     alpha = 1.0  # 用于位置误差指数函数的系数
    #     beta = 1.0   # 速度误差权重
        
    #     # 位置奖励：当位置误差为0时，指数值为1；误差越大，奖励指数值迅速减小
    #     reward_pos = np.exp(-alpha * pos_error**2)
        
    #     # 速度惩罚：这里简单地用速度误差的平方来惩罚
    #     reward_vel = - beta * vel_error**2
        
    #     # 总奖励为位置奖励加上速度惩罚
    #     reward = reward_pos + reward_vel

    #     return reward

    def render(self, mode='human'):
        """
        可视化当前状态（可选）。
        """
        rospy.loginfo(f"Current State: {self.current_state}")

