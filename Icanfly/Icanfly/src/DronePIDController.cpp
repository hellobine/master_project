// DroneController.cpp
#include "pid_controller/DronePIDController.h"

#define RADIAN M_PI / 180.0

int n_loops = 1;
double exec_loop_rate = 100.0;
double circle_velocity = 3.0;
double radius = 8.0;
Eigen::Vector3d circle_center = Eigen::Vector3d(0,0,2);

void DroneController::computeManeuver() {
    
    all_points.clear();  // 清空历史轨迹点

    // 1. ----------------------------------------------
    // Trajectory circle_trajectory =
    // trajectory_generation_helper::circles::computeVerticalCircleTrajectory(
    //       circle_center, 2 ,radius, circle_velocity, M_PI_2,
    //       -(-0.5+2 * n_loops) * M_PI, exec_loop_rate);

    // 2. ----------------------------------------------
    Trajectory circle_trajectory = trajectory_generation_helper::circles::computeHorizontalCircleTrajectory(
        circle_center,radius, circle_velocity, M_PI_2,
        -(-0.5 + 2 * n_loops) * M_PI, exec_loop_rate);        
    trajectory_generation_helper::heading::addForwardHeading(&circle_trajectory);

  std::vector<quadrotor_common::TrajectoryPoint> sampled_trajectory;
  for (auto point : circle_trajectory.points) {
        all_points.push_back(point);

    //    std::printf("bodyrates: %.2f, %.2f, %.2f\n", point.bodyrates.x(),
    //                point.bodyrates.y(), point.bodyrates.z());
  }
}

double convertGlobalYawToLocalYaw(double target_yaw, double current_yaw) {
    // 计算目标 yaw 相对当前 yaw 的偏差
    double local_yaw = target_yaw - current_yaw;

    // 将偏差归一化到 [-π, π]
    if (local_yaw > M_PI) {
        local_yaw -= 2 * M_PI;
    } else if (local_yaw < -M_PI) {
        local_yaw += 2 * M_PI;
    }

    return local_yaw;
}

DroneController::DroneController(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : pid_z_vz(4.0, 0.0, 0.1),
      pid_vz_thrust(2.0, 0.01, 0.1),
      pid_vx_accx(2.0, 0.0, 0.1),
      pid_vy_accy(2.0, 0.0, 0.1),
      pid_y_vy(4.0, 0.0, 0.1),
      pid_x_vx(4.0, 0.0, 0.1),
      pid_yaw_rate(1.0, 0.0, 0.1){


    target_x_=3;
    target_y_=3;
    target_z_=3;

    control_pub_ = nh.advertise<mav_msgs::RollPitchYawrateThrust>(
        rotors_control::kDefaultCommandRollPitchYawrateThrustTopic, 10);
    odometry_sub_ = nh.subscribe("/hummingbird/odometry_sensor1/odometry", 1,
                                 &DroneController::odometryCallback, this);
        
    marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10); // 创建 RViz 发布者

    // yaw_pub = nh.advertise<std_msgs::Float64>("/hummingbird/yaw", 10);
    
    target_point_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/hummingbird/target_point", 10);
    // 定时器，每隔 0.1 秒调用一次发布函数
    // marker_timer = nh.createTimer(ros::Duration(0.1), &DroneController::publishMarkerCallback, this);
    ROS_INFO("Drone Controller initialized!");
}

void DroneController::publishTrajectoryMarkers() {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "computed_trajectory";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::POINTS;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    for (const auto& point : all_points) {
        geometry_msgs::Point p;
        p.x = point.position.x();
        p.y = point.position.y();
        p.z = point.position.z();
        marker.points.push_back(p);
    }

    marker_pub.publish(marker);
}

void DroneController::run() {
    computeManeuver();  // 生成轨迹点

    if (all_points.empty()) {
        ROS_ERROR("No trajectory points! Exiting...");
        return;
    }

    ros::Time start_time = ros::Time::now();
    size_t current_point_idx = 0;

    ros::Rate rate(10);  // 控制频率（20Hz），可以根据实际情况调整
    while (ros::ok()) {
        publishTrajectoryMarkers(); // 仅用于可视化，不影响控制
        
        ros::Time current_time = ros::Time::now();
        ros::Duration elapsed_time = current_time - start_time;

        // **使用二分查找找到最近的轨迹点**
        size_t low = 0, high = all_points.size() - 1;
        while (low < high) {
            size_t mid = (low + high) / 2;
            if (all_points[mid].time_from_start < elapsed_time) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        current_point_idx = low;

        // **避免越界**
        if (current_point_idx >= all_points.size()) {
            ROS_WARN("Trajectory finished. Holding position.");
            // // holdPosition();  // 让无人机停留在最后一个点
            // break;

            start_time = ros::Time::now();
            current_point_idx = 0;
            ROS_INFO("Restarting trajectory...");
            // continue;
        }

        const auto& target_point = all_points[current_point_idx];

        // 计算偏航角（yaw）
        double target_yaw = std::atan2(
            2.0 * (target_point.orientation.w() * target_point.orientation.z() +
                   target_point.orientation.x() * target_point.orientation.y()),
            1.0 - 2.0 * (target_point.orientation.y() * target_point.orientation.y() +
                         target_point.orientation.z() * target_point.orientation.z()));

        // **发送目标点**
        publishTargetPoint(target_point);

        // **发送控制指令**
        controlLogic(target_point.position.x(),
                     target_point.position.y(),
                     target_point.position.z(),
                     target_yaw);

        // // **可选：如果当前点的时间已经过去了，尝试跳到下一个**
        // while (current_point_idx < all_points.size() - 1 &&
        //        all_points[current_point_idx + 1].time_from_start <= elapsed_time) {
        //     current_point_idx++;
        // }

        rate.sleep();
        ros::spinOnce();
    }
}

void DroneController::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
    rotors_control::eigenOdometryFromMsg(msg, &odometry_);
}

void DroneController::controlLogic(double target_x_, double target_y_,
 double target_z_, double target_yaw) {

    static ros::Time last_time = ros::Time::now();
    ros::Time current_time = ros::Time::now();
    double dt = (current_time - last_time).toSec();
    last_time = current_time;
    if (dt <= 0) return;  // 避免异常情况


    double pos_x_cur = odometry_.position.x();
    double pos_y_cur = odometry_.position.y();
    double pos_z_cur = odometry_.position.z();

    double vel_x_cur = odometry_.velocity.x();
    double vel_y_cur = odometry_.velocity.y();
    double vel_z_cur = odometry_.velocity.z();

    Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();
    double yaw = atan2(R(1, 0), R(0, 0));

        // 转换全局 yaw 为局部 yaw
    // double target_local_yaw = convertGlobalYawToLocalYaw(target_yaw, yaw);


    ROS_INFO_STREAM("[ target_yaw: " << target_yaw << ", yaw :" << yaw<< " ]");


    double vel_z_des = pid_z_vz.calculate(target_z_, pos_z_cur, dt);
    // ROS_INFO_STREAM("PID check: [target input: " << target_z_ << ", curr input:" << pos_z_cur << ", out:  " << vel_z_des << "]");
    double thrust_z = pid_vz_thrust.calculate(vel_z_des, vel_z_cur, dt) + 0.68 * 9.81;
    // ROS_INFO_STREAM("thrust_z check: [target input: " << vel_z_des << ", curr input:" << vel_z_cur << ", out:  " << thrust_z << "]");
    
    double vel_x_des = pid_x_vx.calculate(target_x_, pos_x_cur, dt);
    double vel_y_des = pid_y_vy.calculate(target_y_, pos_y_cur, dt);

    double b_vel_x_des = vel_x_des * cos(yaw) + vel_y_des * sin(yaw);
    double b_vel_y_des = -vel_x_des * sin(yaw) + vel_y_des * cos(yaw);

    double b_acc_x_des = pid_vx_accx.calculate(b_vel_x_des, vel_x_cur, dt);
    double b_acc_y_des = pid_vy_accy.calculate(b_vel_y_des, vel_y_cur, dt);

    double yaw_rate = pid_yaw_rate.calculate(target_yaw, yaw, dt);

    double des_pitch = b_acc_x_des*RADIAN;
    double des_roll = -b_acc_y_des*RADIAN;

    mav_msgs::RollPitchYawrateThrust control_msg;
    control_msg.thrust.z = thrust_z;
    control_msg.pitch = des_pitch;
    control_msg.roll = des_roll;
    control_msg.yaw_rate = 0;
    control_pub_.publish(control_msg);


    // ROS_INFO_STREAM("cmd: [" << des_pitch << ", " << des_roll << ", " << thrust_z << "]");
    // ROS_INFO_STREAM("Velocity: [" << vel_x_cur << ", " << vel_y_cur << ", " << vel_z_cur << "]");
    // ROS_INFO_STREAM("pos: [" << pos_x_cur << ", " << pos_y_cur << ", " << pos_z_cur << "]");
    // ROS_INFO_STREAM("yaw: [" << yaw << ", target_local_yaw; " << target_local_yaw << "]");
}

void DroneController::publishTargetPoint(const TrajectoryPoint& target_point) {
    geometry_msgs::PoseStamped target_msg;
    target_msg.header.stamp = ros::Time::now();
    target_msg.header.frame_id = "world";
    target_msg.pose.position.x = target_point.position.x();
    target_msg.pose.position.y = target_point.position.y();
    target_msg.pose.position.z = target_point.position.z();
    target_msg.pose.orientation.w = target_point.orientation.w();
    target_msg.pose.orientation.x = target_point.orientation.x();
    target_msg.pose.orientation.y = target_point.orientation.y();
    target_msg.pose.orientation.z = target_point.orientation.z();
    target_point_pub_.publish(target_msg);
}