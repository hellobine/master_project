cmake_minimum_required(VERSION 2.8.3)
project(Icanfly)

add_definitions(-std=c++14)

#--------------------------#
# Generation of SDF models #
#--------------------------#

# We need Gazebo version >= 3.0.0 to generate iris.sdf file
# (gz sdf ... command needs to be available)
find_package(gazebo REQUIRED)
find_package(PythonInterp REQUIRED)


find_package(catkin REQUIRED COMPONENTS gazebo_msgs geometry_msgs mav_msgs
  roscpp sensor_msgs 
  rotors_control 
  minimum_jerk_trajectories 
  quadrotor_common 
  quadrotor_msgs 
  trajectory_generation_helper 
  rpg_mpc
)

find_package(casadi REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    gazebo_msgs
    geometry_msgs
    mav_msgs
    roscpp
    sensor_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(
    include/mpc_controller
    include/pid_controller
    include/quadrotor_mpc_codegen
    # include ../rpg_mpc/model/quadrotor_mpc_codegen/
    # include../../rotors_control/trajectory_generation_helper
    # include()
    # include/trajectory_generation_helper
    ${catkin_INCLUDE_DIRS}
    ${casadi_INCLUDE_DIRS}
)

# target_include_directories(drone_mpc_controller PUBLIC
#         ../rpg_mpc/model/quadrotor_mpc_codegen/
#         ../rpg_mpc/externals/qpoases
#         ../rpg_mpc/externals/qpoases/INCLUDE
#         ../rpg_mpc/externals/qpoases/SRC
# )

# add_executable(waypoint_publisher src/waypoint_publisher.cpp)
# target_link_libraries(waypoint_publisher ${catkin_LIBRARIES})
# add_dependencies(waypoint_publisher ${catkin_EXPORTED_TARGETS})

# add_executable(waypoint_publisher_file src/waypoint_publisher_file.cpp)
# target_link_libraries(waypoint_publisher_file ${catkin_LIBRARIES})
# add_dependencies(waypoint_publisher_file ${catkin_EXPORTED_TARGETS})

# add_executable(hovering_example src/hovering_example.cpp)
# target_link_libraries(hovering_example ${catkin_LIBRARIES})
# add_dependencies(hovering_example ${catkin_EXPORTED_TARGETS})

# add_executable(run_circle_locus src/run_circle_locus.cpp)
# target_link_libraries(run_circle_locus ${catkin_LIBRARIES})
# add_dependencies(run_circle_locus ${catkin_EXPORTED_TARGETS})

## 1️⃣ 先编译主要逻辑（DronePIDController.cpp）
add_library(PIDController
  src/DronePIDController.cpp
)
add_library(MPCController
  src/DroneMPCController.cpp
)

## 连接 ROS 依赖
target_link_libraries(PIDController MPCController ${catkin_LIBRARIES})
add_dependencies(PIDController MPCController ${catkin_EXPORTED_TARGETS})

add_executable(pid_drone_controller src/DronePIDControllerNode.cpp)
target_link_libraries(pid_drone_controller PIDController ${catkin_LIBRARIES})
add_dependencies(pid_drone_controller ${catkin_EXPORTED_TARGETS})

add_executable(drone_mpc_controller src/DroneMPCControllerNode.cpp)
target_link_libraries(drone_mpc_controller MPCController ${catkin_LIBRARIES} /usr/local/lib/libcasadi.so)
add_dependencies(drone_mpc_controller ${catkin_EXPORTED_TARGETS})




# add_executable(pid_drone_xyz_controller
#     src/DroneXYZControllerNode.cpp
#     src/DroneXYZController.cpp
#     src/mpc_controller.cpp  # 关键！添加此行
# )

# target_link_libraries(pid_drone_xyz_controller   ${CASADI_LIBRARIES} ${catkin_LIBRARIES})
# add_dependencies(pid_drone_xyz_controller ${catkin_EXPORTED_TARGETS})

# target_include_directories(drone_mpc_controller PUBLIC
#         ../../../rpg_mpc/model/quadrotor_mpc_codegen/
#         ../../../rpg_mpc/externals/qpoases
#         ../../../rpg_mpc/externals/qpoases/INCLUDE
#         ../../../rpg_mpc/externals/qpoases/SRC)
# ${CMAKE_SOURCE_DIR}/

# target_include_directories(pid_drone_xyz_controller PUBLIC
#         ../../rotors_control/trajectory_generation_helper/include())

# foreach(dir launch models resource worlds)
#    install(DIRECTORY ${dir}/
#       DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
# endforeach(dir)

# install(TARGETS waypoint_publisher waypoint_publisher_file hovering_example
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
