// DroneController.cpp
#include "rotors_gazebo/DroneXYZController.h"

#define RADIAN M_PI / 180.0

DroneController::DroneController(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : pid_z_vz(1.0, 0.0, 0.1),
      pid_vz_thrust(1.0, 0.0, 0.1),
      pid_vx_accx(1.0, 0.0, 0.1),
      pid_vy_accy(1.0, 0.0, 0.1),
      pid_y_vy(1.0, 0.0, 0.1),
      pid_x_vx(1.0, 0.0, 0.1),
      pid_yaw_rate(1.0, 0.0, 0.1){


    target_x_=3;
    target_y_=3;
    target_z_=3;

    control_pub_ = nh.advertise<mav_msgs::RollPitchYawrateThrust>(
        rotors_control::kDefaultCommandRollPitchYawrateThrustTopic, 10);
    odometry_sub_ = nh.subscribe("/hummingbird/odometry_sensor1/odometry", 1,
                                 &DroneController::odometryCallback, this);
        // 创建 RViz 发布者
    marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);

    ROS_INFO("Drone Controller initialized!");
}

void DroneController::run() {


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
    geometry_msgs::Point p;
    
    for (double t = 0; t <= t_max; t += dt) {
        double angle = fmod(2 * M_PI / 10 * t, 2 * M_PI);  // 10s飞一圈
        TrajectoryPoint point;
        point.position.x() = a * sin(angle);
        point.position.y() = b * sin(2 * angle);  // 8字轨迹
        point.position.z() = 3.0 + c * cos(angle);
        point.yaw = fmod(angle + M_PI / 2, 2 * M_PI);  // 更新偏航角
        trajectory.push_back(point);


        p.x = point.position.x();
        p.y = point.position.y();
        p.z = point.position.z();
        trajectory_marker.points.push_back(p);
    }

    // Default desired position and yaw.
    // Eigen::Vector3d desired_position(trajectory[0].position.x(), 
    //     trajectory[0].position.y(), trajectory[0].position.z());
    // double desired_yaw = trajectory[0].yaw;

    // Overwrite defaults if set as node parameters.
    // nh_private.param("x", desired_position.x(), desired_position.x());
    // nh_private.param("y", desired_position.y(), desired_position.y());
    // nh_private.param("z", desired_position.z(), desired_position.z());
    // nh_private.param("yaw", desired_yaw, desired_yaw);

    int target_index=0;

    ros::Rate rate(10);
    while (ros::ok()) {
        marker_pub.publish(trajectory_marker);
        controlLogic(trajectory[target_index].position.x(), trajectory[target_index].position.y(),
         trajectory[target_index].position.z(), trajectory[target_index].yaw);

        target_index = target_index % trajectory.size() + 1;
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
    double dt = 0.1;
    last_time = current_time;

    // if (dt <= 0) return;

    double pos_x_cur = odometry_.position.x();
    double pos_y_cur = odometry_.position.y();
    double pos_z_cur = odometry_.position.z();

    double vel_x_cur = odometry_.velocity.x();
    double vel_y_cur = odometry_.velocity.y();
    double vel_z_cur = odometry_.velocity.z();



    Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();
    double yaw = atan2(R(1, 0), R(0, 0));

    double vel_z_des = pid_z_vz.calculate(target_z_, pos_z_cur, dt);
    ROS_INFO_STREAM("PID check: [target input: " << target_z_ << ", curr input:" << pos_z_cur << ", out:  " << vel_z_des << "]");
    double thrust_z = pid_vz_thrust.calculate(vel_z_des, vel_z_cur, dt) + 0.68 * 9.81;
    ROS_INFO_STREAM("thrust_z check: [target input: " << vel_z_des << ", curr input:" << vel_z_cur << ", out:  " << thrust_z << "]");
    
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
    control_msg.yaw_rate = yaw_rate;
    control_pub_.publish(control_msg);

    // ROS_INFO_STREAM("cmd: [" << des_pitch << ", " << des_roll << ", " << thrust_z << "]");
    // ROS_INFO_STREAM("Velocity: [" << vel_x_cur << ", " << vel_y_cur << ", " << vel_z_cur << "]");
    // ROS_INFO_STREAM("pos: [" << pos_x_cur << ", " << pos_y_cur << ", " << pos_z_cur << "]");
    ROS_INFO_STREAM("yaw: [" << yaw << ", target_yaw; " << target_yaw << "]");
}
