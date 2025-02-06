#include "mpc_controller/DroneMPCController.h"
#include <std_srvs/Empty.h>
#include <thread>
#include <chrono>
int main(int argc, char** argv) {
    ros::init(argc, argv, "drone_mpc_controller");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
        
    mpc_controller::DroneMpcController controller(nh, private_nh);

    // 创建 MPC 控制器

    ROS_INFO_STREAM("[ I am debug!!!!!!!!!!!!!!!!!!!!!!! ]");

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
    controller.run();
    return 0;
}