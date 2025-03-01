#!/usr/bin/env python
import rospy
import matplotlib.pyplot as plt
import numpy as np
import threading
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf.transformations as tf_trans

# 定义线程锁，保证数据访问安全
state_lock = threading.Lock()

# 用于记录无人机状态的历史数据
drone_time_history = []
drone_x_history = []
drone_y_history = []
drone_z_history = []
drone_yaw_history = []

# 用于记录目标状态的历史数据
ref_time_history = []
ref_x_history = []
ref_y_history = []
ref_z_history = []
ref_yaw_history = []

def quaternion_to_yaw(qx, qy, qz, qw):
    """将四元数转换为偏航角（yaw）"""
    _, _, yaw = tf_trans.euler_from_quaternion([qx, qy, qz, qw])
    return yaw

def odom_callback(msg):
    """订阅 /hummingbird/ground_truth/odometry，记录无人机状态"""
    global drone_time_history, drone_x_history, drone_y_history, drone_z_history, drone_yaw_history
    t = msg.header.stamp.to_sec()
    with state_lock:
        drone_time_history.append(t)
        drone_x_history.append(msg.pose.pose.position.x)
        drone_y_history.append(msg.pose.pose.position.y)
        drone_z_history.append(msg.pose.pose.position.z)
        drone_yaw_history.append(quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ))

def reference_state_callback(msg):
    """订阅 /hummingbird/autopilot/reference_tracking_point，记录目标状态"""
    global ref_time_history, ref_x_history, ref_y_history, ref_z_history, ref_yaw_history
    t = msg.header.stamp.to_sec()
    with state_lock:
        ref_time_history.append(t)
        ref_x_history.append(msg.pose.position.x)
        ref_y_history.append(msg.pose.position.y)
        ref_z_history.append(msg.pose.position.z)
        ref_yaw_history.append(quaternion_to_yaw(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ))

def update_plot(axs):
    """更新图形，显示历史数据"""
    with state_lock:
        t_drone = np.array(drone_time_history)
        x_drone = np.array(drone_x_history)
        y_drone = np.array(drone_y_history)
        z_drone = np.array(drone_z_history)
        yaw_drone = np.array(drone_yaw_history)
        t_ref = np.array(ref_time_history)
        x_ref = np.array(ref_x_history)
        y_ref = np.array(ref_y_history)
        z_ref = np.array(ref_z_history)
        yaw_ref = np.array(ref_yaw_history)
    
    # 更新位置随时间变化的子图
    axs[0].cla()
    axs[0].plot(t_drone, x_drone, label="Drone X", color='b')
    axs[0].plot(t_drone, y_drone, label="Drone Y", color='g')
    axs[0].plot(t_drone, z_drone, label="Drone Z", color='r')
    axs[0].plot(t_ref, x_ref, '--', label="Ref X", color='b')
    axs[0].plot(t_ref, y_ref, '--', label="Ref Y", color='g')
    axs[0].plot(t_ref, z_ref, '--', label="Ref Z", color='r')
    axs[0].set_title("Position vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position")
    axs[0].legend()
    axs[0].grid(True)

    # 更新偏航角随时间变化的子图
    axs[1].cla()
    axs[1].plot(t_drone, yaw_drone, label="Drone Yaw", color='b')
    axs[1].plot(t_ref, yaw_ref, '--', label="Ref Yaw", color='r')
    axs[1].set_title("Yaw vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Yaw (rad)")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.draw()

if __name__ == '__main__':
    rospy.init_node('quadrotor_control_metric', anonymous=True)

    # 订阅无人机状态和目标状态话题
    rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, odom_callback)
    rospy.Subscriber('/hummingbird/autopilot/reference_tracking_point', PoseStamped, reference_state_callback)

    # 开启 Matplotlib 交互模式，并在主线程中初始化图形
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    plt.show()

    rate = rospy.Rate(1.0)  # 设置更新频率为 1 Hz
    while not rospy.is_shutdown():
        update_plot(axs)
        plt.pause(0.001)  # 短暂暂停，保证 GUI 事件处理
        rate.sleep()
