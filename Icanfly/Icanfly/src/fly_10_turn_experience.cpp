#include <ros/ros.h>
#include "autopilot_helper/autopilot_helper.h"
#include <quadrotor_msgs/Trajectory.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <cmath>
#include <rotors_comm/WindSpeed.h>
#include "minimum_jerk_trajectories/RapidTrajectoryGenerator.h"
#include <visualization_msgs/Marker.h>

using namespace minimum_jerk_trajectories;
using namespace autopilot_helper;

int main(int argc, char** argv) {
  ros::init(argc, argv, "fly_10_turn_experience_node");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  autopilot_helper::AutoPilotHelper autopilot_helper(nh, private_nh);

    
  ros::Publisher windspeed_pub = nh.advertise<rotors_comm::WindSpeed>("/hummingbird/wind_speed", 1);
  rotors_comm::WindSpeed msg;

  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "world";
  msg.velocity.x = 10.0;
  msg.velocity.y = 0.0;
  msg.velocity.z = 0.0;


  quadrotor_common::Trajectory traj_msg;
  // autopilot_helper.generateEightTrajectory(traj_msg);
  autopilot_helper.generateCircleTrajectory(traj_msg);
  // autopilot_helper.generateCurveTrajectory(traj_msg);

  int flag=0;


  ros::Rate rate(10); // 每秒检查 10 次
  while (ros::ok()) {

    if (autopilot_helper.getCurrentAutopilotState() == autopilot::States::HOVER) {
        if(flag==0){
          flag+=1;
          autopilot_helper.sendTrajectory(traj_msg);
        }  
    }
    windspeed_pub.publish(msg);
    ros::spinOnce();  
    rate.sleep();
  }
  return 0;
}
