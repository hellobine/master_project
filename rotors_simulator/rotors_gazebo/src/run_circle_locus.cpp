//  run_circle_locus.cpp
// 飞一个圆形轨迹

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
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

struct TrajectoryPoint {
    Eigen::Vector3d position;
    double yaw;
};
// static const std::string kDefaultOdometryTopic =
//     mav_msgs::default_topics::ODOMETRY; // "odometry"

// 当前无人机的位置和偏航角
Eigen::Vector3d current_position(0.0, 0.0, 1.0);
double current_yaw = 0.0;



// 里程计回调函数
void OdometryCallback(const nav_msgs::OdometryConstPtr& msg) {
    // 更新当前位置
    current_position.x() = msg->pose.pose.position.x;
    current_position.y() = msg->pose.pose.position.y;
    current_position.z() = msg->pose.pose.position.z;

    // 更新偏航角（从四元数转换）
    double roll, pitch;
    tf::Quaternion q(msg->pose.pose.orientation.x,
                     msg->pose.pose.orientation.y,
                     msg->pose.pose.orientation.z,
                     msg->pose.pose.orientation.w);
    tf::Matrix3x3(q).getRPY(roll, pitch, current_yaw);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "circle_locus_example");
    ros::NodeHandle nh;
    // Create a private node handle for accessing node parameters.
    ros::NodeHandle nh_private("~");
    ros::Publisher trajectory_pub =
        nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>(
            mav_msgs::default_topics::COMMAND_TRAJECTORY, 10);
    ROS_INFO("Started circle locus example.");

  // 订阅无人机的里程计信息
    ros::Subscriber odometry_sub_ = nh.subscribe(rotors_control::kDefaultOdometryTopic, 1,
                               OdometryCallback);


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

    double begin_t = ros::Time::now().toSec();
    double t = 0;

    // 设置一个阈值，表示当前位置与目标轨迹点之间的距离
    double distance_threshold = 0.0001;  // 例如，距离小于 0.5 米时更新目标轨迹点

    ros::Rate rate(10);
    size_t last_target_index = 0;  // 保存当前选择的目标轨迹点的索引
    size_t target_index = 0;  // 保存当前选择的目标轨迹点的索引

    while (ros::ok())
    {   
                  
        t = ros::Time::now().toSec() - begin_t;

        marker_pub.publish(trajectory_marker);

        // 查找与当前定位正前方1米最接近的轨迹点
        TrajectoryPoint target_point;
        double min_distance = std::numeric_limits<double>::infinity();
        // double lookahead_distance = 1.0;  // 正前方1米


        target_point = trajectory[target_index];



        // // 如果当前位置已经到达目标轨迹点，更新目标点
        // if (min_distance < 0.5 && last_target_index != target_index) {  // 假设阈值为 0.5 米
        //     size_t next_target_index = target_index + 1; // 更新目标为下一个轨迹点
        //     if (next_target_index < trajectory.size()) {
        //         target_point = trajectory[next_target_index];  // 更新目标轨迹点
        //         ROS_INFO("Updated target to point index: %zu, Position: x = %.2f, y = %.2f, z = %.2f",
        //                 next_target_index, target_point.position.x(), target_point.position.y(), target_point.position.z());
        //     }
        // }


        // 更新目标位置和偏航角
        Eigen::Vector3d target_position_(0.0, 0.0, 1.0);
        double target_yaw_ = 0.0;

        // 将目标轨迹点赋给 desired_position 和 desired_yaw
        target_position_.x() = target_point.position.x();
        target_position_.y() = target_point.position.y();
        target_position_.z() = target_point.position.z();

        // 偏航角（假设轨迹中已经有偏航角信息，或根据某种规则计算）
        target_yaw_ = target_point.yaw;  // 使用当前的偏航角，也可以根据轨迹调整偏航
        // 打印目标轨迹点的坐标
        ROS_INFO("Target Position: x = %.2f, y = %.2f, z = %.2f", 
                target_position_.x(), target_position_.y(), target_position_.z());


        trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
        trajectory_msg.header.stamp = ros::Time::now();
        mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(
            target_position_, target_yaw_, &trajectory_msg);
        trajectory_pub.publish(trajectory_msg);
        rate.sleep();
        ros::spinOnce();
        
        target_index+=1;
    }
    
    ros::shutdown();

    return 0;
}
