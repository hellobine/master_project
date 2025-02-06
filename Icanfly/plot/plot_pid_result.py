#!/usr/bin/env python
import rospy
import matplotlib.pyplot as plt
from std_msgs.msg import Float64

# 存储数据
yaw_values = []
time_values = []

def callback(data):
    global yaw_values, time_values
    yaw_values.append(data.data)
    time_values.append(rospy.get_time())

    # 仅保留最近 100 个数据点
    if len(yaw_values) > 100:
        yaw_values.pop(0)
        time_values.pop(0)

    # 实时更新图像
    plt.clf()
    plt.plot(time_values, yaw_values, label="Yaw Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Yaw (rad)")
    plt.legend()
    plt.pause(0.01)

if __name__ == '__main__':
    rospy.init_node('yaw_plotter', anonymous=True)
    rospy.Subscriber("/drone/yaw", Float64, callback)

    plt.ion()  # 开启实时绘图
    plt.show()

    rospy.spin()
