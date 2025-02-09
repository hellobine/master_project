#include <ros/ros.h>
#include "autopilot_helper/autopilot_helper.h"
#include <quadrotor_msgs/Trajectory.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <cmath>

#include "minimum_jerk_trajectories/RapidTrajectoryGenerator.h"
#include <visualization_msgs/Marker.h>

using namespace minimum_jerk_trajectories;
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

  autopilot_helper::AutoPilotHelper autopilot_helper(nh, private_nh);
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

   // 轨迹参数设置
  double T = 0.2;                // 每段轨迹持续时间（秒）
  int num_loops = 10;            // 总共 10 圈
  int num_points_per_loop = 100; // 每圈 100 个 waypoint
  int total_waypoints = num_loops * num_points_per_loop;
  Vec3 gravity = Vec3(0,0,-9.81);//[m/s**2]

  // 生成原始 waypoints（这里构造一个 8 字形立体轨迹）
  std::vector<Vec3> waypoints;
  for (int i = 0; i < total_waypoints; ++i) {
    double t = (double)i / num_points_per_loop * 2 * M_PI;
    double x = 5 * cos(t);
    double y = 5 * sin(2 * t) / 2;
    double z = 3 + sin(t);
    waypoints.push_back(Vec3(x, y, z));

    // 同时添加到 RViz Marker 中显示
    // geometry_msgs::Point rviz_point;
    // rviz_point.x = x;
    // rviz_point.y = y;
    // rviz_point.z = z;
    // marker.points.push_back(rviz_point);
  }

  // 定义唯一的轨迹消息
  quadrotor_msgs::Trajectory traj_msg;
  traj_msg.header.stamp = ros::Time::now();
  traj_msg.type = quadrotor_msgs::Trajectory::GENERAL;

  // 起始状态
  Vec3 pos0 = waypoints[0];
  Vec3 vel0(0, 0, 0);
  Vec3 acc0(0, 0, 0);

  double cumulative_time = 0.0;  // 用于计算每个点的 time_from_start

  // 对每一段 waypoint（从 pos0 到下一个 waypoint）生成 mini-jerk 轨迹，并采样存储中间点,这里每段采样的点数可以根据需要调整，比如设为 10 个采样点
  int num_samples_per_segment = 10;
  for (size_t i = 1; i < waypoints.size(); i++) {
    Vec3 posf = waypoints[i];
    Vec3 velf(0, 0, 0);
    Vec3 accf(0, 0, 0);

    // 生成从 pos0 到 posf 的最小跃度轨迹段
    RapidTrajectoryGenerator traj_segment(pos0, vel0, acc0, gravity);
    traj_segment.SetGoalPosition(posf);
    traj_segment.SetGoalVelocity(velf);
    traj_segment.SetGoalAcceleration(accf);
    traj_segment.Generate(T);

    // 对该轨迹段按固定时间间隔采样
    for (int j = 0; j <= num_samples_per_segment; j++) {
      double t_sample = T * j / double(num_samples_per_segment);
      // 调用 EvaluatePosition( t ) 得到当前时刻的位置信息
      Vec3 pos_sample = traj_segment.GetPosition(t_sample); 

      quadrotor_msgs::TrajectoryPoint point;
      point.time_from_start = ros::Duration(cumulative_time + t_sample);
      point.pose.position.x = pos_sample[0];
      point.pose.position.y = pos_sample[1];
      point.pose.position.z = pos_sample[2];

      geometry_msgs::Point rviz_point;
      rviz_point.x = pos_sample[0];
      rviz_point.y = pos_sample[1];
      rviz_point.z = pos_sample[2];
      marker.points.push_back(rviz_point);

      // 计算 heading
      if (j < num_samples_per_segment) {
        // 采样点未到末尾：使用下一个采样点计算方向
        Vec3 next_sample = traj_segment.GetPosition(t_sample + (T / num_samples_per_segment));
        point.heading = computeHeadingToNextPoint(pos_sample[0], pos_sample[1],
                                                   next_sample[0], next_sample[1]);
      } else {
        // 当前采样点为该段最后一点：若还有后续 waypoint，则用当前点和下一段第一个采样点计算 heading
        if (i < waypoints.size() - 1) {
          // 这里简单使用当前点与下一个 waypoint 计算 heading
          point.heading = computeHeadingToNextPoint(pos_sample[0], pos_sample[1],
                                                     waypoints[i+1][0], waypoints[i+1][1]);
        } else {
          // 最后一个点，直接使用前一个点的 heading（若存在），否则置 0
          if (!traj_msg.points.empty())
            point.heading = traj_msg.points.back().heading;
          else
            point.heading = 0;
        }
      }

      traj_msg.points.push_back(point);
    }
    // 更新累计时间，并将本段终点作为下段起点
    cumulative_time += T;
    pos0 = posf;
    vel0 = velf;
    acc0 = accf;
  }

  // 发布轨迹到 RViz
  marker_pub.publish(marker);

  // 等待 autopilot 进入 HOVER 状态
  ros::Rate rate(10); // 每秒检查 10 次
  while (ros::ok()) {

    if (autopilot_helper.getCurrentAutopilotState() == autopilot::States::HOVER) {
        ROS_INFO("Autopilot is now in HOVER state. Proceeding...");
        autopilot_helper.sendTrajectory(traj_msg);
        ROS_INFO("10 圈 8 字形立体轨迹已发送");
    }
    ros::spinOnce();  // **保持 ROS 事件循环，防止节点退出**
    rate.sleep();
  }
  return 0;
}
