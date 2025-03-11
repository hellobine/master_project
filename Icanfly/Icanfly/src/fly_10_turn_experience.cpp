#include <ros/ros.h>
#include "autopilot_helper/autopilot_helper.h"
#include <quadrotor_msgs/Trajectory.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <cmath>
#include <visualization_msgs/Marker.h>

#include "minimum_jerk_trajectories/RapidTrajectoryGenerator.h"
#include <visualization_msgs/Marker.h>
using namespace minimum_jerk_trajectories;

using namespace autopilot_helper;



int main(int argc, char** argv) {
  ros::init(argc, argv, "fly_10_turn_experience_node");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  // 定义用于 RViz 可视化的 Marker 发布器
  ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("RL/sendReferenceState", 10);
  // 构造 RViz Marker 消息
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";  // 根据实际使用的坐标系调整
  marker.header.stamp = ros::Time::now();
  marker.ns = "trajectory_points";
  //  marker.id = point_index;  // 使用点的索引作为 id
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;

  // 设置 Marker 的大小
  marker.scale.x = 0.2;
  marker.scale.y = 0.2;
  marker.scale.z = 0.2;
  
  // 设置颜色 (RGBA)
  marker.color.r = 1.0f;
  marker.color.g = 0.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0f;

  
  // 初始化 AutoPilotHelper
  autopilot_helper::AutoPilotHelper autopilot_helper(nh, private_nh);

  quadrotor_common::Trajectory traj_msg;
  // autopilot_helper.generateEightTrajectory(traj_msg);
  // autopilot_helper.generateCircleTrajectory(traj_msg);
  autopilot_helper.generateCurveTrajectory(traj_msg);

  // 定义圆参数
  double radius = 3.0;          // 圆半径
  int num_points = 1000;           // 圆上的固定点数
  double center_x = 0.0;        // 圆心横坐标
  double center_y = 0.0;        // 圆心纵坐标
  double z = 3.0;               // 固定高度

  std::vector<quadrotor_common::TrajectoryPoint> circle_points;
  for (int i = 0; i < num_points; ++i)
  {
    double angle = 2 * M_PI * i / num_points;  // 均分圆周
    quadrotor_common::TrajectoryPoint point;
    point.position.x() = center_x + radius * cos(angle);
    point.position.y() = center_y + radius * sin(angle);
    point.position.z() = z;   // 设置固定高度

    circle_points.push_back(point);

  }


  size_t point_index = 0;

  ros::Rate rate(10); // 每秒检查 10 次
  while (ros::ok()) {
    // if (autopilot_helper.getCurrentAutopilotState() == autopilot::States::HOVER) {
    //     if(flag==0){
    //       flag+=1;
    //       autopilot_helper.sendTrajectory(traj_msg);
    //     }  
    // }
    ros::spinOnce();  

    // 如果还有未发送的点，则每次循环发送一个点
    if (point_index < circle_points.size())
    {
      // Marker 的位置与当前点保持一致
      marker.pose.position.x = circle_points[point_index].position.x();
      marker.pose.position.y = circle_points[point_index].position.y();
      marker.pose.position.z = circle_points[point_index].position.z();
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;
          // 发布 Marker
      marker_pub.publish(marker);

      autopilot_helper.sendReferenceState(circle_points[point_index]);
      point_index++;
      if(point_index==circle_points.size()){
        point_index=0;
      }
    }

    rate.sleep();
  }
  return 0;
}
