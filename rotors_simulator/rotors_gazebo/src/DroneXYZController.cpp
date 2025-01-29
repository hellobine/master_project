// DroneController.cpp
#include "rotors_gazebo/DroneXYZController.h"

#define RADIAN M_PI / 180.0



void DroneController::computeManeuver() {
    
    all_points.clear();  // 清空历史轨迹点

    // 生成初始轨迹（示例：直线）
    TrajectoryPoint start_state;
    start_state.position = Eigen::Vector3d(0, 0, 0);  // 初始位置
    start_state.velocity = Eigen::Vector3d(0, 0, 0);   // 初始速度


    // 创建轨迹序列对象
    fpv_aggressive_trajectories::AcrobaticSequence sequence(start_state);
    Eigen::Vector3d target_pos1(5.0, 6.0, 2.0);
    Eigen::Vector3d target_vel1(1.0, 0, 0);
    bool success1 = sequence.appendStraight(target_pos1, target_vel1, 0.0, 1.0, 100.0);

    Eigen::Vector3d target_pos2(0.0, 0.0, 0.0);
    Eigen::Vector3d target_vel2(0.0, 0, 0);

    bool success2 = sequence.appendStraight(target_pos2, target_vel2, 0.0, 1.0, 100.0);

    if (success2 && success1) {
        // 提取所有轨迹点
        auto trajectory_list = sequence.getManeuverList();
        for (const auto& trajectory : trajectory_list) {
            for (const auto& point : trajectory.points) {
                all_points.push_back(point);
            }
        }

        ROS_INFO("Generated %lu trajectory points.", all_points.size());
    } else {
        ROS_ERROR("Failed to generate trajectory!");
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

// void DroneController::publishMarkerCallback(const ros::TimerEvent&) {
//     // 初始化Marker消息
//     // visualization_msgs::Marker trajectory_marker;
//     // trajectory_marker.header.frame_id = "world"; // 设定坐标系
//     // trajectory_marker.header.stamp = ros::Time::now();
//     // trajectory_marker.ns = "trajectory";
//     // trajectory_marker.id = 0;  // 设置 ID
//     // trajectory_marker.type = visualization_msgs::Marker::ARROW; // 用于显示方向
//     // trajectory_marker.action = visualization_msgs::Marker::ADD;
//     // trajectory_marker.scale.x = 0.2; // 箭头长度
//     // trajectory_marker.scale.y = 0.05; // 箭头宽度
//     // trajectory_marker.scale.z = 0.05; // 箭头高度
//     // trajectory_marker.color.a = 1.0f;  // 不透明

//     // 生成完整的轨迹
//     std::vector<TrajectoryPoint> trajectory;
//     double a = 5.0, b = 3.0, c = 1.0;
//     double t_max = 30.0; // 轨迹的持续时间（秒）
//     double dt = 0.1; // 轨迹点的时间间隔

//     for (double t = 0; t <= t_max; t += dt) {
//         double angle = fmod(2 * M_PI / 10 * t, 2 * M_PI);  // 10s飞一圈
//         TrajectoryPoint point;
//         point.position.x() = a * sin(angle);
//         point.position.y() = b * sin(2 * angle);  // 8字轨迹
//         point.position.z() = 3.0 + c * cos(angle);

//         // 如果不是最后一个点，计算指向下一个点的 yaw
//         if (t + dt <= t_max) {
//             double next_angle = fmod(2 * M_PI / 10 * (t + dt), 2 * M_PI);  // 下一个点的角度
//             double next_x = a * sin(next_angle);
//             double next_y = b * sin(2 * next_angle);

//             // 计算当前点指向下一个点的 yaw
//             double dx = next_x - point.position.x();
//             double dy = next_y - point.position.y();
//             point.yaw = atan2(dy, dx);
//         } else {
//             // 对于最后一个点，保持与前一个点的 yaw 一致（平滑）
//             point.yaw = trajectory.back().yaw;
//         }

//         trajectory.push_back(point);

//         // 创建Marker消息
//         visualization_msgs::Marker marker;
//         marker.header.frame_id = "world";
//         marker.header.stamp = ros::Time::now();
//         marker.ns = "trajectory_arrows";
//         marker.id = t / dt;  // 唯一ID
//         marker.type = visualization_msgs::Marker::ARROW;
//         marker.action = visualization_msgs::Marker::ADD;

//         // 设置箭头尺寸
//         marker.scale.x = 0.2; // 箭头长度
//         marker.scale.y = 0.05; // 箭头宽度
//         marker.scale.z = 0.05; // 箭头高度

//         // 设置箭头位置
//         marker.pose.position.x = point.position.x();
//         marker.pose.position.y = point.position.y();
//         marker.pose.position.z = point.position.z();

//         // 计算方向四元数
//         tf2::Quaternion quaternion;
//         quaternion.setRPY(0, 0, point.yaw); // 偏航角旋转
//         marker.pose.orientation.x = quaternion.x();
//         marker.pose.orientation.y = quaternion.y();
//         marker.pose.orientation.z = quaternion.z();
//         marker.pose.orientation.w = quaternion.w();

//         // 设置颜色（可根据 yaw 的值动态设置颜色）
//         marker.color.r = fabs(sin(point.yaw));  // 红色分量与 yaw 的正弦相关
//         marker.color.g = fabs(cos(point.yaw));  // 绿色分量与 yaw 的余弦相关
//         marker.color.b = 1.0 - marker.color.r;  // 蓝色与红色互补
//         marker.color.a = 1.0;  // 不透明

//         // 发布Marker
//         marker_pub.publish(marker);

//     }
// }

// void DroneController::run() {

//     // 初始化Marker消息
//     visualization_msgs::Marker trajectory_marker;
//     trajectory_marker.header.frame_id = "world"; // 设定坐标系
//     trajectory_marker.header.stamp = ros::Time::now();
//     trajectory_marker.ns = "trajectory";
//     trajectory_marker.id = 0;  // 设置 ID
//     trajectory_marker.type = visualization_msgs::Marker::POINTS; // 用于连接点的线条
//     trajectory_marker.action = visualization_msgs::Marker::ADD;
//     trajectory_marker.scale.x = 0.05; // 轨迹线的宽度
//     trajectory_marker.color.r = 1.0f;  // 设置颜色（红色）
//     trajectory_marker.color.g = 0.0f;
//     trajectory_marker.color.b = 0.0f;
//     trajectory_marker.color.a = 1.0f;  // 不透明

//      // 生成完整的轨迹
//     std::vector<TrajectoryPoint> trajectory;
//     double a = 5.0, b = 3.0, c = 1.0;
//     double t_max = 30.0; // 轨迹的持续时间（秒）
//     double dt = 0.1; // 轨迹点的时间间隔
//     geometry_msgs::Point p;
    
//     for (double t = 0; t <= t_max; t += dt) {
        
//         double angle = fmod(2 * M_PI / 10 * t, 2 * M_PI);  // 10s飞一圈
//         TrajectoryPoint point;
//         point.position.x() = a * sin(angle);
//         point.position.y() = b * sin(2 * angle);  // 8字轨迹
//         point.position.z() = 3.0 + c * cos(angle);

//         // 如果不是最后一个点，计算指向下一个点的 yaw
//         if (t + dt <= t_max) {
//             double next_angle = fmod(2 * M_PI / 10 * (t + dt), 2 * M_PI);  // 下一个点的角度
//             double next_x = a * sin(next_angle);
//             double next_y = b * sin(2 * next_angle);

//             // 计算当前点指向下一个点的 yaw
//             double dx = next_x - point.position.x();
//             double dy = next_y - point.position.y();
//             point.yaw = atan2(dy, dx);
//         } else {
//             // 对于最后一个点，保持与前一个点的 yaw 一致（平滑）
//             point.yaw = trajectory.back().yaw;
//         }

//         trajectory.push_back(point);

//         // // 计算方向四元数
//         // tf2::Quaternion quaternion;
//         // quaternion.setRPY(0, 0, point.yaw); // 偏航角旋转
//         // geometry_msgs::Quaternion orientation;
//         // orientation.x = quaternion.x();
//         // orientation.y = quaternion.y();
//         // orientation.z = quaternion.z();
//         // orientation.w = quaternion.w();

//         // p.x = point.position.x();
//         // p.y = point.position.y();
//         // p.z = point.position.z();
//         // trajectory_marker.points.push_back(p);
//     }

//     // Default desired position and yaw.
//     // Eigen::Vector3d desired_position(trajectory[0].position.x(), 
//     //     trajectory[0].position.y(), trajectory[0].position.z());
//     // double desired_yaw = trajectory[0].yaw;

//     // Overwrite defaults if set as node parameters.
//     // nh_private.param("x", desired_position.x(), desired_position.x());
//     // nh_private.param("y", desired_position.y(), desired_position.y());
//     // nh_private.param("z", desired_position.z(), desired_position.z());
//     // nh_private.param("yaw", desired_yaw, desired_yaw);

//     int target_index=0;

//     ros::Rate rate(10);
//     while (ros::ok()) {
//         // marker_pub.publish(trajectory_marker);

//         controlLogic(trajectory[target_index].position.x(), trajectory[target_index].position.y(),
//         trajectory[target_index].position.z(), trajectory[target_index].yaw);

//         target_index = target_index % trajectory.size() + 1;

//         computeManeuver();
//         rate.sleep();
//         ros::spinOnce();
//     }
// }



void DroneController::run() {
    computeManeuver();  // 生成轨迹点
    
    if (all_points.empty()) {
        ROS_ERROR("No trajectory points! Exiting...");
        return;
    }

    ros::Time start_time = ros::Time::now();
    size_t current_point_idx = 0;

    ros::Rate rate(100);  // 100Hz控制频率
    while (ros::ok()) {
        publishTrajectoryMarkers();
        ros::Time current_time = ros::Time::now();
        ros::Duration elapsed_time = current_time - start_time;

        // 寻找当前时间对应的轨迹点
        while (current_point_idx < all_points.size() && 
               all_points[current_point_idx].time_from_start <= elapsed_time) 
        {
            const auto& target_point = all_points[current_point_idx];
            
            // 发送控制指令
            controlLogic(target_point.position.x(),
                        target_point.position.y(),
                        target_point.position.z(),
                        0);
            
            current_point_idx++;
        }

        // // 轨迹循环逻辑（可选）
        // if (current_point_idx >= all_points.size()) {
        //     start_time = ros::Time::now();
        //     current_point_idx = 0;
        //     ROS_INFO("Restarting trajectory...");
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

        // 转换全局 yaw 为局部 yaw
    double target_local_yaw = convertGlobalYawToLocalYaw(target_yaw, yaw);

    // target_yaw = 


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

    double yaw_rate = pid_yaw_rate.calculate(target_local_yaw, yaw, dt);

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
    ROS_INFO_STREAM("yaw: [" << yaw << ", target_local_yaw; " << target_local_yaw << "]");
}
