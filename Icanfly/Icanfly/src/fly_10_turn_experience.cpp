#include <ros/ros.h>
#include "autopilot_helper/autopilot_helper.h"
#include <quadrotor_msgs/Trajectory.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <cmath>

#include "minimum_jerk_trajectories/RapidTrajectoryGenerator.h"
#include <visualization_msgs/Marker.h>
using namespace minimum_jerk_trajectories;

using namespace autopilot_helper;



int main(int argc, char** argv) {
  ros::init(argc, argv, "fly_10_turn_experience_node");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // 初始化 AutoPilotHelper
  autopilot_helper::AutoPilotHelper autopilot_helper(nh, private_nh);


  quadrotor_common::Trajectory traj_msg;
  // autopilot_helper.generateEightTrajectory(traj_msg);
  // autopilot_helper.generateCircleTrajectory(traj_msg);
  autopilot_helper.generateCurveTrajectory(traj_msg);

  int flag=0;


  // 等待 autopilot 进入 HOVER 状态
  ros::Rate rate(10); // 每秒检查 10 次
  while (ros::ok()) {

    // if (autopilot_helper.getCurrentAutopilotState() == autopilot::States::HOVER) {
        // if(flag==0){
          // flag+=1;
          autopilot_helper.sendTrajectory(traj_msg);
        // }  
    // }
    ros::spinOnce();  // **保持 ROS 事件循环，防止节点退出**
    rate.sleep();
  }
  return 0;
}
