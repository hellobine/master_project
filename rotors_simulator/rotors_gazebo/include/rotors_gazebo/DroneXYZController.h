#ifndef DRONE_CONTROLLER_H
#define DRONE_CONTROLLER_H

#include <ros/ros.h>
#include <mav_msgs/RollPitchYawrateThrust.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Core>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <mav_msgs/conversions.h>
#include <mav_msgs/default_topics.h>
#include <std_srvs/Empty.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_datatypes.h>
#include "rotors_control/common.h"
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <vector>


struct TrajectoryPoint {
    Eigen::Vector3d position;
    double yaw;
};

// PID 控制器类
class PID {
public:
    PID(double kp, double ki, double kd)
        : kp_(kp), ki_(ki), kd_(kd), prev_error_(0), integral_(0) {}

    double calculate(double setpoint, double current, double dt) {
        if (dt <= 0) {
            std::cerr << "Warning: dt should be positive." << std::endl;
            return 0;
        }

        double error = setpoint - current;
        integral_ += error * dt;
        double derivative = (error - prev_error_) / dt;
        prev_error_ = error;

        return kp_ * error + ki_ * integral_ + kd_ * derivative;
    }

    void reset() {
        integral_ = 0;
        prev_error_ = 0;
    }

    void setGains(double kp, double ki, double kd) {
        kp_ = kp;
        ki_ = ki;
        kd_ = kd;
    }

    void getGains(double& kp, double& ki, double& kd) const {
        kp = kp_;
        ki = ki_;
        kd = kd_;
    }

private:
    double kp_, ki_, kd_;
    double prev_error_;
    double integral_;
};

class DroneController {
public:
    DroneController(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    void run();

    // 公开的 PID 控制器成员
    PID pid_z_vz, pid_vz_thrust;
    PID pid_vx_accx, pid_vy_accy;
    PID pid_y_vy, pid_x_vx;
    PID pid_yaw_rate;

private:
    rotors_control::EigenOdometry odometry_;
    double target_x_, target_y_, target_z_;

    ros::Publisher control_pub_;
    ros::Subscriber odometry_sub_;
    ros::Publisher marker_pub;
    ros::Timer marker_timer;

    void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
    void controlLogic(double target_x_, double target_y_, 
        double target_z_, double target_yaw);

    void publishMarkerCallback(const ros::TimerEvent&);
};

#endif // DRONE_CONTROLLER_H
