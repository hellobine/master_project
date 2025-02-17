#ifndef LQR_H
#define LQR_H

#include <ros/ros.h>
// #include <rosflight_msgs/Command.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <dynamic_reconfigure/server.h>
#include <lqr_controller/Figure8Config.h>

#include <quadrotor_common/control_command.h>
#include <quadrotor_common/quad_state_estimate.h>
#include <quadrotor_common/trajectory.h>
#include <quadrotor_common/trajectory_point.h>

#include <memory>
#include <Eigen/Core>
#include <std_msgs/Float64MultiArray.h>
#include <Eigen/Dense>
#include <iostream>
#include "lin_alg_tools/care.h"
#include "geometry/quat.h"

#include "lqr_controller/figure8.h"
#include "lqr_controller/waypoints.h"
#include "lqr_controller/logger.h"
#include "lqr_controller/lqr_controller_params.h"

enum : int {
  xPOS = 0,
  xATT = 3,
  xVEL = 7,
  xOMEGA = 10,
  xZ = 13
};

enum : int {
  dxPOS = 0,
  dxVEL = 3,
  dxATT = 6,
  dxZ = 9
};

enum : int {
  uTHROTTLE = 0,
  uOMEGA = 1,
  uZ = 4,
};

typedef Eigen::Matrix<double, xZ, 1> StateVector;
typedef Eigen::Matrix<double, dxZ, 1> ErrStateVector;
typedef Eigen::Matrix<double, uZ, 1> InputVector;
typedef Eigen::Matrix<double, dxZ, dxZ> ErrStateErrStateMatrix;
typedef Eigen::Matrix<double, dxZ, uZ> ErrStateInputMatrix;
typedef Eigen::Matrix<double, uZ, dxZ> InputErrStateMatrix;
typedef Eigen::Matrix<double, uZ, uZ> InputInputMatrix;

namespace lqr
{

class LQRController
{
  public:
    LQRController();
    void solverCore(const StateVector &x, const StateVector &x_c,
                        const InputVector &ur, InputVector &u);

    quadrotor_common::ControlCommand off();

    quadrotor_common::ControlCommand run(
      const quadrotor_common::QuadStateEstimate& state_estimate,
      const quadrotor_common::Trajectory& reference_trajectory,
      const LQRControllerParams& config);
    
    Eigen::VectorXd quadrotorDynamics(const Eigen::VectorXd &x, const Eigen::VectorXd &u, double dt);

    void finiteDifferenceDynamics(const Eigen::VectorXd &x, const Eigen::VectorXd &u, double dt,
      Eigen::MatrixXd &A, Eigen::MatrixXd &B);
    
    double runningCost(const Eigen::VectorXd &x, const Eigen::VectorXd &u,
        const Eigen::VectorXd &x_ref, const Eigen::VectorXd &u_ref,
        const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R);

    double finalCost(const Eigen::VectorXd &x, const Eigen::VectorXd &x_ref,
      const Eigen::MatrixXd &Qf);    

    void runningCostDerivatives(const Eigen::VectorXd &x, const Eigen::VectorXd &u,
      const Eigen::VectorXd &x_ref, const Eigen::VectorXd &u_ref,
      const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
      Eigen::VectorXd &l_x, Eigen::VectorXd &l_u,
      Eigen::MatrixXd &l_xx, Eigen::MatrixXd &l_uu,
      Eigen::MatrixXd &l_ux);

    void finalCostDerivatives(const Eigen::VectorXd &x, const Eigen::VectorXd &x_ref,
      const Eigen::MatrixXd &Qf,
      Eigen::VectorXd &l_x, Eigen::MatrixXd &l_xx);

    void iLQR(const Eigen::VectorXd &x0, std::vector<Eigen::VectorXd> &u_seq, int horizon, double dt,
      const std::vector<Eigen::VectorXd> &x_ref_traj, const std::vector<Eigen::VectorXd> &u_ref_traj,
      const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &Qf,
      std::vector<Eigen::VectorXd> &x_seq_out, std::vector<Eigen::VectorXd> &u_seq_out);

  private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    const int state_dim = 10;   // [p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z]
    const int control_dim = 4;  // [T, w_x, w_y, w_z]
    const double g_z = 9.8066;  
 
    MatrixXd A_, B_, Q_, R_, P_, K_;

    const double grav_val_ = 9.8;
    double hover_throttle_=0.7; // mass = 0.716
    double drag_const_;

    double max_pos_err_;
    double max_alt_err_;
    double max_ang_err_;
    double max_vel_err_;
    double max_throttle_err_;
    double max_omega_err_;

    double max_throttle_c_;
    double min_throttle_c_;
    double max_omega_c_;
    double min_omega_c_;

    void saturateInput(Eigen::VectorXd &u);
    void saturateErrorVec(Eigen::Vector3d &err, double max_err);
    void saturateErrorVec(Eigen::Vector3d &err, double max_err, double max_err2);
    void computeControl(const quadrotor_common::QuadStateEstimate& state_estimate,
      const quadrotor_common::TrajectoryPoint& reference_state,
      quadrotor_common::ControlCommand* command);
    
    VectorXd constructStateVector(
        const quadrotor_common::QuadStateEstimate& state);

    double computeReferenceThrust(
          const quadrotor_common::TrajectoryPoint& ref);

    VectorXd constructReferenceVector(const quadrotor_common::TrajectoryPoint& ref);
};

}
#endif /* LQR_H */
