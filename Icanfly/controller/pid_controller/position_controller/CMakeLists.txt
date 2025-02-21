# cmake_minimum_required(VERSION 2.8.3)
# project(pid_controller)

# ## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
# add_compile_options(-O3)

# find_package(catkin_simple REQUIRED)
# catkin_simple(ALL_DEPS_REQUIRED)

# cs_add_library(${PROJECT_NAME} src/autopilot_pid_instance.cpp)


# cs_add_executable(autopilot_pid_instance
# src/autopilot_pid_instance.cpp)




# cs_install()
# cs_export()


cmake_minimum_required(VERSION 3.0.2)
project(pid_controller)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  autopilot
)


catkin_package(
  INCLUDE_DIRS include
#   LIBRARIES position_controller
  CATKIN_DEPENDS roscpp
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


add_library(${PROJECT_NAME}
#   src/autopilot_pid_instance.cpp
  src/position_controller.cpp
)

add_executable(pid_controller_node
  src/autopilot_pid_instance.cpp
)
add_dependencies(pid_controller autopilot ${catkin_EXPORTED_TARGETS})

target_link_libraries(pid_controller_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)


# install(TARGETS ${PROJECT_NAME} position_controller
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )


install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
