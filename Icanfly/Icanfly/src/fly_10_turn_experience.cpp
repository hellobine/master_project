#include <ros/ros.h>
#include "autopilot_helper/autopilot_helper.h"
#include <quadrotor_msgs/Trajectory.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <cmath>
#include <visualization_msgs/Marker.h>

using namespace autopilot_helper;
ros::Publisher marker_pub;
// 角度转换：弧度 → 角度
#define RAD2DEG(x) ((x) * 180.0 / M_PI)

// 计算 heading（偏航角）指向下一个轨迹点
double computeHeadingToNextPoint(double x1, double y1, double x2, double y2) {
    return atan2(y2 - y1, x2 - x1);  // 计算相对偏航角
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "fly_10_turn_experience_node");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // 初始化 AutoPilotHelper
  autopilot_helper::AutoPilotHelper autopilot_helper(nh, private_nh);

  // 创建 RViz 轨迹发布者
  marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1, true);
 
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";  // 适配你的坐标系
  marker.header.stamp = ros::Time::now();
  marker.ns = "trajectory";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::POINTS;  // **修改为点**
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = 0.1;  // **点的大小**
  marker.scale.y = 0.1;
  marker.color.r = 1.0;  // 红色
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;
  marker.pose.orientation.w = 1.0;

  // 轨迹参数
  double A = 3.0;  // 8 字形的 x 轴振幅
  double B = 2.0;  // 8 字形的 y 轴振幅
  double C = 1.0;  // 高度变化幅度
  double T = 0.1;  // 轨迹时间间隔
  int num_loops = 10;  // 总共 10 圈
  int num_points_per_loop = 1000;  // 每圈 100 个点
  int total_points = num_loops * num_points_per_loop;

  // 生成 10 圈 8 字形立体轨迹
  quadrotor_msgs::Trajectory traj_msg;
  traj_msg.header.stamp = ros::Time::now();
  traj_msg.type = quadrotor_msgs::Trajectory::GENERAL;

  for (int i = 0; i < total_points; ++i) {
    double t = (double)i / num_points_per_loop * 2 * M_PI;  // 让 t 从 0 到 20π（10 圈）

    quadrotor_msgs::TrajectoryPoint point;
    point.time_from_start = ros::Duration(T * i);

    // 8 字形立体轨迹
    point.pose.position.x = 5 * sin(t);
    point.pose.position.y = 3 * sin(2 * t);
    point.pose.position.z = 3 + C * sin(3 * t);  // z 方向上下波动

    traj_msg.points.push_back(point);
     // 在 RViz 可视化轨迹点
    geometry_msgs::Point rviz_point;
    rviz_point.x = point.pose.position.x;
    rviz_point.y = point.pose.position.y;
    rviz_point.z = point.pose.position.z;
    marker.points.push_back(rviz_point);
  }
    // **计算 heading 使其指向下一个轨迹点**
  for (int i = 0; i < total_points; ++i) {
    if (i < total_points - 1) {
      traj_msg.points[i].heading = computeHeadingToNextPoint(
          traj_msg.points[i].pose.position.x, traj_msg.points[i].pose.position.y,
          traj_msg.points[i + 1].pose.position.x, traj_msg.points[i + 1].pose.position.y);
    } else {
      // 最后一个点的 heading 设为倒数第二个点的 heading
      traj_msg.points[i].heading = traj_msg.points[i - 1].heading;
    }
  }

  // 发布轨迹到 RViz
  marker_pub.publish(marker);

  // 等待 autopilot 进入 HOVER 状态
  ros::Rate rate(10); // 每秒检查 10 次
  while (ros::ok()) {

    if (autopilot_helper.getCurrentAutopilotState() == autopilot::States::HOVER) {
        ROS_INFO("Autopilot is now in HOVER state. Proceeding...");
        autopilot_helper.sendTrajectory(traj_msg);
        ROS_INFO("10 圈 8 字形立体轨迹已发送，共 %d 个点", total_points);
    }
    ros::spinOnce();  // **保持 ROS 事件循环，防止节点退出**
    rate.sleep();
  }
  return 0;
}
