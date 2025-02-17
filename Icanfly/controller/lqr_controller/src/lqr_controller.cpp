#include "lqr_controller/lqr_controller.h"
#include "lqr_controller/eigen.h"

#include "geometry/quat.h"
#include "geometry/support.h"

#include "lqr_controller/logger.h"
#include <Eigen/Eigenvalues>
#include <stdexcept>
#include <chrono>
using namespace std::chrono;

namespace lqr
{

MatrixXd solveCARE(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& R)
{
    int n = A.rows();
    // 计算 R 的逆
    MatrixXd R_inv = R.inverse();

    // 构造 Hamiltonian 矩阵 H (2n x 2n)
    MatrixXd H(2 * n, 2 * n);
    H.topLeftCorner(n, n) = A;
    H.topRightCorner(n, n) = -B * R_inv * B.transpose();
    H.bottomLeftCorner(n, n) = -Q;
    H.bottomRightCorner(n, n) = -A.transpose();

    // 求解 H 的特征值和特征向量
    ComplexEigenSolver<MatrixXd> ces(H);
    MatrixXcd eigVec = ces.eigenvectors();

    // 将稳定子空间的特征向量分离出来（特征值实部 < 0）
    MatrixXcd X(n, n);
    MatrixXcd Y(n, n);
    int count = 0;
    for (int i = 0; i < 2 * n; i++) {
        if (ces.eigenvalues()[i].real() < 0 && count < n) {
            X.col(count) = eigVec.block(0, i, n, 1);
            Y.col(count) = eigVec.block(n, i, n, 1);
            count++;
        }
    }
    // 计算 P = real(Y * X^{-1})
    MatrixXcd P_complex = Y * X.inverse();
    MatrixXd P = P_complex.real();
    return P;
}
  


LQRController::LQRController() :
  nh_(ros::NodeHandle()),
  nh_private_("~")
{
  nh_private_.getParam("lqr_controller/hover_throttle", hover_throttle_);
  nh_private_.getParam("lqr_controller/drag_constant", drag_const_);
  nh_private_.getParam("lqr_controller/lqr_max_pos_error", max_pos_err_);
  nh_private_.getParam("lqr_controller/lqr_max_alt_error", max_alt_err_);
  nh_private_.getParam("lqr_controller/lqr_max_vel_error", max_vel_err_);
  nh_private_.getParam("lqr_controller/lqr_max_ang_error", max_ang_err_);
  //nh_private_.getParam("lqr_max_throttle_error", max_throttle_err_);
  //nh_private_.getParam("lqr_max_omega_error", max_omega_err_);
  nh_private_.getParam("lqr_controller/lqr_max_throttle_command", max_throttle_c_);
  nh_private_.getParam("lqr_controller/lqr_min_throttle_command", min_throttle_c_);
  nh_private_.getParam("lqr_controller/lqr_max_omega_command", max_omega_c_);
  nh_private_.getParam("lqr_controller/lqr_min_omega_command", min_omega_c_);



  // ErrStateVector q_diag;
  // lqr::importMatrixFromParamServer(nh_private_, q_diag, "lqr_controller/Q");
  // Q_ = q_diag.asDiagonal();

  // InputVector r_diag;
  // lqr::importMatrixFromParamServer(nh_private_, r_diag, "lqr_controller/R");
  // R_ = r_diag.asDiagonal();


 
  //----------------------------------------------------------------------
  // 系统状态维数和控制输入维数
  n_ = 10; // 状态向量: [p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z]
  m_ = 4;  // 控制输入: [T, w_x, w_y, w_z]

  // // 定义线性模型 A, B（基于之前的线性化结果）
  // A_ = MatrixXd::Zero(n_, n_);
  // // 位置状态 p_x, p_y, p_z 的导数与速度 v_x, v_y, v_z 相关
  // A_(0, 7) = 1;
  // A_(1, 8) = 1;
  // A_(2, 9) = 1;
  // // 速度动态：仅考虑非零项来自于姿态的线性化
  // // 例如：dot(v_x) 对 q_y 的偏导为 2*g，其中 g = 9.8066
  // double g = 9.8066;
  // A_(7, 5) = 2 * g;  // ∂(v_x)/∂(q_y)
  // A_(8, 4) = -2 * g; // ∂(v_y)/∂(q_x)
  // // 其他状态（如四元数部分）在平衡点处的偏导均为 0

  // // 定义 B 矩阵
  // B_ = MatrixXd::Zero(n_, m_);
  // // 姿态四元数更新：dot(q_x) = 0.5*w_x, dot(q_y) = 0.5*w_y, dot(q_z) = 0.5*w_z
  // // 注意：四元数顺序 [q_w, q_x, q_y, q_z]，这里仅考虑 q_x, q_y, q_z 更新
  // B_(4, 1) = 0.5; // 对应 w_x
  // B_(5, 2) = 0.5; // 对应 w_y
  // B_(6, 3) = 0.5; // 对应 w_z
  // // 推力 T 对速度 v_z 的影响：dot(v_z) = T - g
  // B_(9, 0) = 1.0; // 对应 T

  // // 定义 LQR 的权重矩阵 Q 和 R
  Q_ = MatrixXd::Zero(n_, n_);
  // 根据设计要求，例如对位置赋较高权重：Q = diag(100, 100, 100, 10, 10, 10, 10, 1, 1, 1)
  Q_(0, 0) = 100;
  Q_(1, 1) = 100;
  Q_(2, 2) = 100;
  Q_(3, 3) = 10;  // q_w（虽在四元数中但可适当设置）
  Q_(4, 4) = 10;
  Q_(5, 5) = 10;
  Q_(6, 6) = 10;
  Q_(7, 7) = 1;
  Q_(8, 8) = 1;
  Q_(9, 9) = 1;

  R_ = MatrixXd::Identity(m_, m_); // 控制输入代价

  // // 求解 CARE 得到 P 矩阵
  // P_ = solveCARE(A_, B_, Q_, R_);
  // // 计算反馈增益矩阵 K = R^{-1} B^T P
  // K_ = R_.inverse() * B_.transpose() * P_;

  // // 设置平衡点（悬停状态）x0
  x0_ = VectorXd::Zero(n_);
  // // 对应 [p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z]
  // // 悬停时位置和速度为 0，四元数取 [1, 0, 0, 0]
  x0_(0) = 0.0;  // p_x = 0
  x0_(1) = 0.0;  // p_y = 0
  x0_(2) = 5.0;  // p_z = 3
  x0_(3) = 1.0;  // q_w = 1


}


VectorXd LQRController::constructStateVector(
  const quadrotor_common::QuadStateEstimate& state) {
  VectorXd x(n_);
  x << state.position.x(), state.position.y(), state.position.z(),
      state.orientation.w(), state.orientation.x(),
      state.orientation.y(), state.orientation.z(),
      state.velocity.x(), state.velocity.y(), state.velocity.z();
  return x;
}

double LQRController::computeReferenceThrust(
    const quadrotor_common::TrajectoryPoint& ref) {
  // 根据参考加速度计算所需推力
  const double des_acc_z = ref.acceleration.z() + 9.81;
  Eigen::Quaterniond q(ref.orientation.w(), ref.orientation.x(),
                      ref.orientation.y(), ref.orientation.z());
  q.normalize();
  const double thrust_coeff = q.w()*q.w() - q.x()*q.x() 
                            - q.y()*q.y() + q.z()*q.z();
  return des_acc_z / thrust_coeff;
}

VectorXd LQRController::constructReferenceVector(
  const quadrotor_common::TrajectoryPoint& ref) 
{
VectorXd x_ref(n_);

// 位置状态 [p_x, p_y, p_z]
x_ref(0) = ref.position.x();
x_ref(1) = ref.position.y();
x_ref(2) = ref.position.z();

// 四元数 [q_w, q_x, q_y, q_z]
x_ref(3) = ref.orientation.w();  // q_w
x_ref(4) = ref.orientation.x();  // q_x
x_ref(5) = ref.orientation.y();  // q_y
x_ref(6) = ref.orientation.z();  // q_z

// 速度状态 [v_x, v_y, v_z]
x_ref(7) = ref.velocity.x();
x_ref(8) = ref.velocity.y();
x_ref(9) = ref.velocity.z();

return x_ref;
} 

void LQRController::saturateInput(Eigen::VectorXd &u)
{
  if (u(0) > max_throttle_c_)
    u(0) = max_throttle_c_;
  else if (u(0) < min_throttle_c_)
    u(0) = min_throttle_c_;

  for (int i = 1; i < 4; i++)
  {
    if (u(i) > max_omega_c_)
      u(i) = max_omega_c_;
    else if (u(i) < min_omega_c_)
      u(i) = min_omega_c_;
  }
}

void LQRController::saturateErrorVec(Eigen::Vector3d& err, double max_err)
{
  for (int i = 0; i < 3; i++)
  {
    if (std::abs(err(i)) > max_err)
      err(i) = err(i) / std::abs(err(i)) * max_err;
  }
}

void LQRController::saturateErrorVec(Eigen::Vector3d &err, double max_err,
                                     double max_err2)
{
  for (int i = 0; i < 2; i++)
  {
    if (std::abs(err(i)) > max_err)
      err(i) = err(i) / std::abs(err(i)) * max_err;
  }
  int i = 2;
  if (std::abs(err(i)) > max_err2)
    err(i) = err(i) / std::abs(err(i)) * max_err2;
}


quadrotor_common::ControlCommand LQRController::off() {
  quadrotor_common::ControlCommand command;
  command.armed = false; // 明确设置armed为false以关闭电机
  // command.control_mode = quadrotor_common::ControlCommand::ControlMode::NONE;
  command.collective_thrust = 0.0;
  command.bodyrates.setZero();
  return command;
}

quadrotor_common::ControlCommand LQRController::run(
    const quadrotor_common::QuadStateEstimate& state_estimate,
    const quadrotor_common::Trajectory& reference_trajectory,
    const LQRControllerParams& config) {
  quadrotor_common::ControlCommand command;
  command.armed = true;
  command.control_mode = quadrotor_common::ControlMode::BODY_RATES;
  // 处理空轨迹情况，返回off命令
  if (reference_trajectory.points.empty()) {
    ROS_WARN("Empty trajectory provided. Returning off command.");
    return off();
  }

  // 获取当前时间对应的参考点（简化示例，假设轨迹点按时间排序）
  const auto& reference_state = reference_trajectory.points.front(); // 更复杂的时间查找逻辑可在此处添加

  // // 打印参考状态（这里同时打印了 Trajectory 的一些信息）
  // ROS_INFO_STREAM("参考轨迹时间戳: " << reference_trajectory.timestamp);
  // ROS_INFO_STREAM("轨迹类型: " << static_cast<int>(reference_trajectory.trajectory_type));
  // std::cout << "reference_state.position.x: " << std::endl << reference_state << std::endl;

  computeControl(state_estimate, reference_state, &command);
  return command;
}

void LQRController::computeControl(
  const quadrotor_common::QuadStateEstimate& state_estimate,
  const quadrotor_common::TrajectoryPoint& reference_state,
  quadrotor_common::ControlCommand* command)
{

  //   ROS_INFO_STREAM("参考状态："
  //     << "\n 位置: [" << reference_state.position.x() << ", "
  //                      << reference_state.position.y() << ", "
  //                      << reference_state.position.z() << "]");

  // 提取参考四元数并归一化
  Eigen::Quaterniond q_ref(
    reference_state.orientation.w(),
    reference_state.orientation.x(),
    reference_state.orientation.y(),
    reference_state.orientation.z());
  q_ref.normalize();
  

  // 根据参考状态重新计算A矩阵
  MatrixXd A = MatrixXd::Zero(n_, n_);
  // 位置导数部分保持不变
  A(0, 7) = 1;
  A(1, 8) = 1;
  A(2, 9) = 1;
  
  // 计算推力方向相关项
  const double T_ref = computeReferenceThrust(reference_state); // 需实现推力估计
  const double qw = q_ref.w(), qx = q_ref.x(), qy = q_ref.y(), qz = q_ref.z();

  // 速度动态线性化 (基于当前参考姿态)
  A(7, 3) =  2*T_ref*qy; // ∂vx/∂qw
  A(7, 4) =  2*T_ref*qz; // ∂vx/∂qx
  A(7, 5) =  2*T_ref*qw; // ∂vx/∂qy 
  A(7, 6) =  2*T_ref*qx; // ∂vx/∂qz

  A(8, 3) = -2*T_ref*qx; // ∂vy/∂qw
  A(8, 4) = -2*T_ref*qw; // ∂vy/∂qx
  A(8, 5) =  2*T_ref*qz; // ∂vy/∂qy
  A(8, 6) =  2*T_ref*qy; // ∂vy/∂qz

  // A(9, 3) =  2*T_ref*qz; // ∂vz/∂qw
  // A(9, 4) = -2*T_ref*qx; // ∂vz/∂qx
  // A(9, 5) = -2*T_ref*qy; // ∂vz/∂qy
  // A(9, 6) =  2*T_ref*qw; // ∂vz/∂qz

  A(9, 3) =  0.0;
  A(9, 4) = -4 * T_ref * qx;
  A(9, 5) = -4 * T_ref * qy;
  A(9, 6) =  0.0;

  // 重新构造B矩阵
  MatrixXd B = MatrixXd::Zero(n_, m_);
  const double thrust_coeff = qw*qw - qx*qx - qy*qy + qz*qz;
  B(9, 0) = thrust_coeff;  // 推力系数随姿态变化

  // 姿态控制通道保持不变
  B(4, 1) = 0.5;
  B(5, 2) = 0.5;
  B(6, 3) = 0.5;

  // 在线求解Riccati方程
  // MatrixXd P = solveCARE(A, B, Q_, R_);
  MatrixXd P;
  try {
     P = solveCARE(A, B, Q_, R_);
  } catch (const std::exception& e) {
      ROS_WARN_THROTTLE(1.0, "Online CARE failed: %s", e.what());
      P = P_; // 使用上一次有效解
  }
  
  ROS_INFO_STREAM("Matrix P:\n" << P);
  MatrixXd K = R_.inverse() * B.transpose() * P;

  // 构造状态误差
  VectorXd x_current = constructStateVector(state_estimate);
  VectorXd x_reference = constructReferenceVector(reference_state);
  VectorXd error = x_current - x_reference;

  // 对误差进行限幅保护
  // 1. 对位置误差（前三个分量）限幅
  {
    Eigen::Vector3d pos_err = error.segment<3>(0);
    saturateErrorVec(pos_err, max_pos_err_); // max_pos_err_ 为位置误差限幅阈值
    error.segment<3>(0) = pos_err;
  }

  // 2. 对姿态误差（4个分量）限幅  
  // 注意：直接对四元数差值进行限幅并不是最佳处理方法，通常需要转换为角误差（例如使用对数映射）
  // 这里仅作简单示例，对每个分量分别限幅 max_ang_err_
  {
    Eigen::Vector4d att_err = error.segment<4>(3);
    for (int i = 0; i < 4; i++) {
      if (std::abs(att_err(i)) > max_ang_err_)
        att_err(i) = att_err(i) / std::abs(att_err(i)) * max_ang_err_;
    }
    error.segment<4>(3) = att_err;
  }

  // 3. 对速度误差（后三个分量）限幅
  {
    Eigen::Vector3d vel_err = error.segment<3>(7);
    saturateErrorVec(vel_err, max_vel_err_);
    error.segment<3>(7) = vel_err;
  }


  VectorXd u_eq(m_);
  u_eq << T_ref, 0, 0, 0;  // 假设角速率补偿为0

  // LQR控制输入加入前馈项： u = u_eq - K * error
  VectorXd u = u_eq - K * error;

  // 计算控制输入
  // VectorXd u = -K * error;

  // 饱和处理
  // u(0) = std::clamp(u(0), 0.0, 15.0);//MAX_THRUST
  ROS_INFO_STREAM("Computed control input u: " << u.transpose());

  saturateInput(u);

  // ROS_INFO_STREAM("Computed control input u: " << u.transpose());

  // 输出映射
  command->collective_thrust = u(0);
  command->bodyrates = u.segment<3>(1);
}

}

