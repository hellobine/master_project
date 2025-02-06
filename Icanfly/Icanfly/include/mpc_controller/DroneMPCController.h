#ifndef MPC_CONTROLLER_H
#define MPC_CONTROLLER_H

#include <ros/ros.h>
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/RollPitchYawrateThrust.h>

#include <visualization_msgs/Marker.h>
#include "rotors_control/common.h"
#include "quadrotor_common/trajectory.h"
#include "quadrotor_msgs/Trajectory.h"
#include <cmath>
#include <mav_msgs/conversions.h>
#include <mav_msgs/default_topics.h>
#include <std_srvs/Empty.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_datatypes.h>
#include "rotors_control/common.h"
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <vector>
#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Dense>
#include "minimum_jerk_trajectories/RapidTrajectoryGenerator.h"
#include <memory>
#include "trajectory_generation_helper/acrobatic_sequence.h"
#include "trajectory_generation_helper/circle_trajectory_helper.h"
#include "trajectory_generation_helper/polynomial_trajectory_helper.h"
#include "trajectory_generation_helper/heading_trajectory_helper.h"
#include "rpg_mpc/mpc_controller.h"
#include "rpg_mpc/mpc_params.h"


// #include <quadrotor_common/quadrotor_control.h>
using namespace quadrotor_common;

namespace mpc_controller {

class MPC {
public:
    MPC(double T, double dt);
    Eigen::Vector4d solve(const Eigen::VectorXd& current_state,
                         const Eigen::MatrixXd& ref_trajectory);
    int getHorizon() const { return N_; }

private:


    void initDynamics();
    casadi::Function createQuadDynamics();
    casadi::MX rk4Integrator(const casadi::MX& x, const casadi::MX& u);
    
    // MPC parameters
    int N_=10, state_dim_ = 10, ctrl_dim_ = 4;
    double dt_=0.1, gz_ = 9.81;
    casadi::Function dynamics_;
    casadi::Opti opti_;
    casadi::MX X_, U_;
    
    // Cost matrices
    Eigen::MatrixXd Q_goal_, Q_pen_, Q_u_;
    casadi::DM Q_goal_mx_, Q_pen_mx_, Q_u_mx_;
    
    // Constraints
    double w_max_yaw_ =3.0, w_max_xy_ =3.0;
    double thrust_min_ = 2.0, thrust_max_ = 10.0;
};

class DroneMpcController {
public:
    DroneMpcController(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    void run();
    
private:
    void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
    void computeManeuver();
    void publishTrajectoryMarkers(const ros::TimerEvent& event);
    Eigen::VectorXd odomToState(const rotors_control::EigenOdometry& odom);
    
    ros::Publisher control_pub_;
    ros::Subscriber odometry_sub_;
    ros::Publisher marker_pub_;
    ros::Publisher target_point_pub_;
    ros::Timer timer_;
    
    // MPC 参数
    rpg_mpc::MpcController<double> mpc_controller_ = rpg_mpc::MpcController<double>(ros::NodeHandle(), ros::NodeHandle("~"),
                                     "rpg_mpc_controller");
    rpg_mpc::MpcParams<double> mpc_params;
    rotors_control::EigenOdometry odometry_;

    std::vector<quadrotor_common::TrajectoryPoint> all_points;
      quadrotor_common::QuadStateEstimate received_state_est_;
    // MPC parameters
    double T_ = 1.0;
    double dt_ = 0.1;
    std::unique_ptr<MPC> mpc_ = std::make_unique<MPC>(T_, dt_);

};

} // namespace mpc_controller

#endif