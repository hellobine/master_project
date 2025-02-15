import gym
import numpy as np
import rospy
from gym import spaces
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand, Trajectory
# nav_msgs::Odometry::ConstPtr& msg
from nav_msgs.msg import Odometry
class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()
        rospy.init_node("drone_rl_env", anonymous=True)

       # 订阅无人机状态估计
        self.state_sub = rospy.Subscriber("hummingbird/ground_truth/odometry", Odometry, self.state_callback)
        self.trajectory_sub = rospy.Subscriber("hummingbird/autopilot/trajectory", Trajectory, self.trajectory_callback)
        self.cmd_pub = rospy.Publisher("/autopilot/velocity_command", ControlCommand, queue_size=1)

        # 观测空间 (位置, 速度, 姿态, 角速度)
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, 0, -2, -2, -2, -1, -1, -1, -1, -2, -2, -2, -5, -5, 0, -2, -2, -2]),  
            high=np.array([5, 5, 3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 5, 5, 3, 2, 2, 2]),
            dtype=np.float32
        )

       # 动作空间 (推力, 机体角速度 X, Y, Z)
        self.action_space = spaces.Box(
            low=np.array([0.5, -2.0, -2.0, -1.0]),  
            high=np.array([2.0, 2.0, 2.0, 1.0]),  
            dtype=np.float32
        )

        self.state = np.zeros(19)  # 包含目标位置
        self.trajectory = []  # 存储轨迹
        self.done = False

    def state_callback(self, msg):
        """更新观测值"""
        self.state[:13] = np.array([
            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,  # 3 维  (位置)
            msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,  # 3 维  (速度)
            msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,  # 4 维 (姿态)
            msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z  # 3 维  (角速度)
        ])

        # print(msg.pose.pose.position.z)


        # 计算当前时间对应的轨迹点
        ref_position, ref_velocity = self.get_reference_trajectory_point()

        self.state[13:16] = ref_position
        self.state[16:] = ref_velocity


        # 终止条件
        if self.state[2] < 0.1:
            self.done = True

    def step(self, action):
        """执行强化学习动作"""
        cmd = ControlCommand()
        # cmd.timestamp = rospy.Time.now()
        cmd.armed = True
        cmd.control_mode = 1  # BODY_RATES 模式
        cmd.collective_thrust = action[0]
        cmd.bodyrates.x = action[1]
        cmd.bodyrates.y = action[2]
        cmd.bodyrates.z = action[3]
        self.cmd_pub.publish(cmd)

        rospy.sleep(0.1)  # 适当的时间步延迟

        # 计算误差
        position_error = np.linalg.norm(self.state[:3] - self.state[13:16])  # 当前位置 - 目标位置
        velocity_error = np.linalg.norm(self.state[3:6] - self.state[16:])  # 当前速度 - 目标速度
        angular_rate_error = np.linalg.norm(self.state[10:])  
        thrust_cost = np.abs(action[0] - 1.0)  

        if self.state[2] < 0.1:
            self.done = True
            crash_penalty = -100.0  
        else:
            crash_penalty = 0.0

        # 代价函数
        cost = (
            10.0 * position_error +  
            5.0 * velocity_error +  
            2.0 * angular_rate_error +  
            1.0 * thrust_cost +  
            crash_penalty  
        )

        reward = -cost  

        return self.state, reward, self.done, {}


    def trajectory_callback(self, msg):
        """订阅 `/autopilot/trajectory` 并存储轨迹"""
        self.trajectory = []
        start_time = msg.header.stamp.to_sec()  # 获取轨迹起始时间
        for point in msg.points:
            time_stamp = start_time + point.time_from_start.to_sec()
            self.trajectory.append({
                "time": time_stamp,
                "position": np.array([point.position.x, point.position.y, point.position.z]),
                "velocity": np.array([point.velocity.x, point.velocity.y, point.velocity.z])
            })
            print(point)
        
    def get_reference_trajectory_point(self):
            """根据当前时间获取最接近的轨迹点"""
            if not self.trajectory:
                return np.zeros(3), np.zeros(3)  # 没有轨迹数据

            current_time = rospy.Time.now().to_sec()
            closest_point = min(self.trajectory, key=lambda p: abs(p["time"] - current_time))
            return closest_point["position"], closest_point["velocity"]

    def reset(self):
        """重置环境"""
        self.done = False
        self.state = np.zeros(19)  # 确保维度与 `observation_space` 匹配
        return self.state.copy()


    def seed(self, seed=None):
        """设置 Gym 环境的随机种子（Stable-Baselines3 兼容）"""
        np.random.seed(seed)
