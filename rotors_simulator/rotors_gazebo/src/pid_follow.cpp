#include <thread>
#include <chrono>
#include <cmath>
#include <Eigen/Core>
#include <mav_msgs/conversions.h>
#include <mav_msgs/default_topics.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_datatypes.h>
#include "rotors_control/common.h"
#include <geometry_msgs/Twist.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <mav_msgs/RollPitchYawrateThrust.h>
#include <cmath>
#include <limits>

struct PID {
    double kp;
    double ki;
    double kd;
    double prev_error;
    double integral;

    PID(double p, double i, double d) : kp(p), ki(i), kd(d), prev_error(0), integral(0) {}

    double calculate(double setpoint, double current, double dt) {
        double error = setpoint - current;
        integral += error * dt;
        double derivative = (error - prev_error) / dt;
        prev_error = error;
        return kp * error + ki * integral + kd * derivative;
    }
};

// Control variables for Outer Loop (Position Control)
PID pid_x(3.0, 0.01, 0.1); // PID for x-axis
PID pid_y(3.0, 0.01, 0.1); // PID for y-axis
PID pid_z(3.5, 0.01, 0.2); // PID for z-axis

// Control variables for Inner Loop (Attitude Control)
PID pid_roll(2.0, 0.0, 0.1);   // PID for roll
PID pid_pitch(2.0, 0.0, 0.1);  // PID for pitch
PID pid_yaw(3.5, 0.0, 0.1);    // PID for yaw

struct TrajectoryPoint {
    Eigen::Vector3d position;
    double yaw;
};

Eigen::Vector3d current_position(0.0, 0.0, 1.0);
double current_roll = 0.0;
double current_pitch = 0.0;
double current_yaw = 0.0;

void OdometryCallback(const nav_msgs::OdometryConstPtr& msg) {
    current_position.x() = msg->pose.pose.position.x;
    current_position.y() = msg->pose.pose.position.y;
    current_position.z() = msg->pose.pose.position.z;

    tf::Quaternion q(msg->pose.pose.orientation.x,
                     msg->pose.pose.orientation.y,
                     msg->pose.pose.orientation.z,
                     msg->pose.pose.orientation.w);
    tf::Matrix3x3(q).getRPY(current_roll, current_pitch, current_yaw);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pid_drone_controller11");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    ros::Publisher control_pub = nh.advertise<mav_msgs::RollPitchYawrateThrust>(rotors_control::kDefaultCommandRollPitchYawrateThrustTopic, 10);
    ros::Subscriber odometry_sub = nh.subscribe(rotors_control::kDefaultOdometryTopic, 10, OdometryCallback);

    ROS_INFO(".............pid pid...............");
    //rviz
    // 创建 RViz 发布者
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);

    // 初始化Marker消息
    visualization_msgs::Marker trajectory_marker;
    trajectory_marker.header.frame_id = "world"; // 设定坐标系
    trajectory_marker.header.stamp = ros::Time::now();
    trajectory_marker.ns = "trajectory";
    trajectory_marker.id = 0;  // 设置 ID
    trajectory_marker.type = visualization_msgs::Marker::LINE_STRIP; // 用于连接点的线条
    trajectory_marker.action = visualization_msgs::Marker::ADD;
    trajectory_marker.scale.x = 0.05; // 轨迹线的宽度
    trajectory_marker.color.r = 1.0f;  // 设置颜色（红色）
    trajectory_marker.color.g = 0.0f;
    trajectory_marker.color.b = 0.0f;
    trajectory_marker.color.a = 1.0f;  // 不透明

     // 生成完整的轨迹
    std::vector<TrajectoryPoint> trajectory;
    double a = 5.0, b = 3.0, c = 1.0;
    double t_max = 30.0; // 轨迹的持续时间（秒）
    double dt = 0.1; // 轨迹点的时间间隔
    for (double t = 0; t <= t_max; t += dt) {
        double angle = fmod(2 * M_PI / 10 * t, 2 * M_PI);  // 10s飞一圈
        TrajectoryPoint point;
        point.position.x() = a * sin(angle);
        point.position.y() = b * sin(2 * angle);  // 8字轨迹
        point.position.z() = 3.0 + c * cos(angle);
        point.yaw = fmod(angle + M_PI / 2, 2 * M_PI);  // 更新偏航角
        trajectory.push_back(point);
    }

    // Default desired position and yaw.
    Eigen::Vector3d desired_position(trajectory[0].position.x(), 
        trajectory[0].position.y(), trajectory[0].position.z());
    double desired_yaw = trajectory[0].yaw;

    // Overwrite defaults if set as node parameters.
    nh_private.param("x", desired_position.x(), desired_position.x());
    nh_private.param("y", desired_position.y(), desired_position.y());
    nh_private.param("z", desired_position.z(), desired_position.z());
    nh_private.param("yaw", desired_yaw, desired_yaw);




    // 轨迹点在Marker中的发布
    for (const auto& point : trajectory) {
        geometry_msgs::Point p;
        p.x = point.position.x();
        p.y = point.position.y();
        p.z = point.position.z();
        trajectory_marker.points.push_back(p);
    }




    std_srvs::Empty srv;
    bool unpaused = ros::service::call("/gazebo/unpause_physics", srv);
    unsigned int i = 0;

    // Trying to unpause Gazebo for 10 seconds.
    while (i <= 10 && !unpaused) {
          ROS_INFO("Wait for 1 second before trying to unpause Gazebo again.");
          std::this_thread::sleep_for(std::chrono::seconds(1));
          unpaused = ros::service::call("/gazebo/unpause_physics", srv);
          ++i;
    }

    if (!unpaused) {
          ROS_FATAL("Could not wake up Gazebo.");
    return -1;
    } else {
          ROS_INFO("Unpaused the Gazebo simulation.");
    }

    // Wait for 5 seconds to let the Gazebo GUI show up.
    ros::Duration(5.0).sleep();

    size_t target_index = 0;
    ros::Rate rate(10);

    while (ros::ok()) {
        

        double dt = 1.0 / 50.0;
        TrajectoryPoint target_point = trajectory[target_index];

        double roll_cmd = pid_y.calculate(target_point.position.y(), current_position.y(), dt);
        double pitch_cmd = pid_x.calculate(target_point.position.x(), current_position.x(), dt);
        double thrust_cmd = pid_z.calculate(target_point.position.z(), current_position.z(), dt);

     //    // Use current roll and pitch for stabilization
     //    double roll_rate = pid_roll.calculate(roll_cmd, 0, dt);
     //    double pitch_rate = pid_pitch.calculate(pitch_cmd, 0, dt);
     //    double yaw_rate = pid_yaw.calculate(target_point.yaw, current_yaw, dt);

        mav_msgs::RollPitchYawrateThrust control_msg;
        control_msg.roll = roll_cmd;
        control_msg.pitch = pitch_cmd;
        control_msg.yaw_rate = target_point.yaw;
        control_msg.thrust.z = thrust_cmd +7;

          // ROS_INFO("Target cmd: roll_rate = %.2f, pitch_rate = %.2f, yaw_rate = %.2f , thrust=%.2f", 
          //       roll_rate, pitch_rate, yaw_rate, thrust_cmd);



        control_pub.publish(control_msg);

        if (++target_index >= trajectory.size()) {
            target_index = 0;
        }else{
          target_index+=1;
        }

        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}
