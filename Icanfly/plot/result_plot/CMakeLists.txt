cmake_minimum_required(VERSION 3.0.2)
project(quadrotor_result_plot)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  gazebo_ros
  quadrotor_msgs
  gazebo_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs nav_msgs 
  sensor_msgs gazebo_ros quadrotor_msgs gazebo_msgs
)

# Python 代码不需要 C++ 编译，但需要安装脚本
catkin_install_python(PROGRAMS
  scripts/plot_result.py


  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(${catkin_INCLUDE_DIRS})
