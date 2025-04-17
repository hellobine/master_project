#include <ros/ros.h>
#include "autopilot_helper/autopilot_helper.h"
#include <quadrotor_msgs/Trajectory.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <cmath>
#include <visualization_msgs/Marker.h>
#include <rotors_comm/WindSpeed.h>

#include "minimum_jerk_trajectories/RapidTrajectoryGenerator.h"
#include <visualization_msgs/Marker.h>
using namespace minimum_jerk_trajectories;

using namespace autopilot_helper;
// from rotors_comm.msg import WindSpeed


int main(int argc, char** argv) {
  ros::init(argc, argv, "fly_10_turn_experience_node");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)
  
  
  // ros::Publisher windspeed_pub = nh.advertise<rotors_comm::WindSpeed>("/hummingbird1/wind_speed", 1);
  // rotors_comm::WindSpeed msg;

  // // 填充消息，例如设置header和velocity的各个字段
  // msg.header.stamp = ros::Time::now();
  // msg.header.frame_id = "world";
  // msg.velocity.x = 0.0;
  // msg.velocity.y = 0.0;

  // msg.velocity.z = 100.0;

  
  autopilot_helper::AutoPilotHelper autopilot_helper(nh, private_nh);

  quadrotor_common::Trajectory traj_msg;

  autopilot_helper.generateEightTrajectory(traj_msg);
  // autopilot_helper.generateCircleTrajectory(traj_msg);
  // autopilot_helper.generateEightFigureTrajectory(traj_msg);
  


  size_t point_index = 0;
  int flag = 0;

  bool tl_training = true;

  ros::Rate rate(10); 
  while (ros::ok()) {
    if (tl_training or (autopilot_helper.getCurrentAutopilotState() == autopilot::States::HOVER)) {
          autopilot_helper.sendTrajectory(traj_msg);
    }
    // windspeed_pub.publish(msg);
    ros::spinOnce();  
    rate.sleep();
  }

  return 0;
}
