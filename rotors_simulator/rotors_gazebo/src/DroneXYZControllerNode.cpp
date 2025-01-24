#include "rotors_gazebo/DroneXYZController.h"

#include <thread>
#include <chrono>
#include <std_srvs/Empty.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "pid_drone_xyz_controller");
    ros::NodeHandle nh;          // 公共 NodeHandle
    ros::NodeHandle private_nh("~"); // 私有 NodeHandle

    // 创建控制器对象
    DroneController drone_controller(nh, private_nh);

    // 动态调整 PID 参数
    drone_controller.pid_z_vz.setGains(2.2, 0.08, 0.2);
    drone_controller.pid_vz_thrust.setGains(2.2, 0.08, 0.2);
    drone_controller.pid_vx_accx.setGains(6.0, 0.01, 0.1);
    drone_controller.pid_vy_accy.setGains(6.0, 0.01, 0.1);
    drone_controller.pid_y_vy.setGains(1, 0, 0.1);
    drone_controller.pid_x_vx.setGains(1, 0, 0.1);

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

    // 运行控制器
    drone_controller.run();

    return 0;
}