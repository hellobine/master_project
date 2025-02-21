#include "lqr_controller/lqr_controller.h"
#include "lqr_controller/eigen.h"
#include "geometry/quat.h"
#include "geometry/support.h"
#include "lqr_controller/logger.h"
#include <Eigen/Eigenvalues>
#include <stdexcept>
#include <chrono>
#include <list>

using namespace std::chrono;

namespace lqr
{
/*****************************  iLQR 相关函数  *****************************/
/**
 * @brief 四旋翼动力学离散化（欧拉积分）
 */
Eigen::VectorXd LQRController::quadrotorDynamics(const Eigen::VectorXd &x, const Eigen::VectorXd &u, double dt)
{
    Eigen::VectorXd x_dot(state_dim);
    // 状态分解
    double p_x = x(0), p_y = x(1), p_z = x(2);
    double q_w = x(3), q_x = x(4), q_y = x(5), q_z = x(6);
    double v_x = x(7), v_y = x(8), v_z = x(9);
    double T   = u(0), wx = u(1), wy = u(2), wz = u(3);

    // 位置导数：dp/dt = v
    x_dot(0) = v_x;
    x_dot(1) = v_y;
    x_dot(2) = v_z;
    // 四元数导数：dot(q) = 0.5 * quat_mult([0, wx, wy, wz], q)
    x_dot(3) = 0.5 * (-wx * q_x - wy * q_y - wz * q_z);
    x_dot(4) = 0.5 * ( wx * q_w + wz * q_y - wy * q_z);
    x_dot(5) = 0.5 * ( wy * q_w - wz * q_x + wx * q_z);
    x_dot(6) = 0.5 * ( wz * q_w + wy * q_x - wx * q_y);
    // 速度导数
    x_dot(7) = 2.0 * (q_w * q_y + q_x * q_z) * T;
    x_dot(8) = 2.0 * (q_y * q_z - q_w * q_x) * T;
    x_dot(9) = (1.0 - 2.0 * q_x * q_x - 2.0 * q_y * q_y) * T - g_z;

    Eigen::VectorXd x_next = x + dt * x_dot;

    // 对四元数部分归一化（防止数值漂移）
    Eigen::Vector4d q_next = x_next.segment(3, 4);
    q_next.normalize();
    x_next.segment(3, 4) = q_next;

    return x_next;
}

/**
 * @brief 利用有限差分计算动力学线性化的雅可比矩阵 A 和 B
 */
void LQRController::finiteDifferenceDynamics(const Eigen::VectorXd &x, const Eigen::VectorXd &u, double dt,
                              Eigen::MatrixXd &A, Eigen::MatrixXd &B)
{
    double eps = 1e-5;
    A = Eigen::MatrixXd::Zero(state_dim, state_dim);
    B = Eigen::MatrixXd::Zero(state_dim, control_dim);
    Eigen::VectorXd f0 = quadrotorDynamics(x, u, dt);
    // 对状态求偏导，计算 A = ∂f/∂x
    for (int i = 0; i < state_dim; i++) {
        Eigen::VectorXd x_perturb = x;
        x_perturb(i) += eps;
        Eigen::VectorXd f_perturb = quadrotorDynamics(x_perturb, u, dt);
        A.col(i) = (f_perturb - f0) / eps;
    }
    // 对控制求偏导，计算 B = ∂f/∂u
    for (int i = 0; i < control_dim; i++) {
        Eigen::VectorXd u_perturb = u;
        u_perturb(i) += eps;
        Eigen::VectorXd f_perturb = quadrotorDynamics(x, u_perturb, dt);
        B.col(i) = (f_perturb - f0) / eps;
    }
}

/**
 * @brief 运行代价函数：l(x,u) = (x - x_ref)^T Q (x - x_ref) + (u - u_ref)^T R (u - u_ref)
 */
double LQRController::runningCost(const Eigen::VectorXd &x, const Eigen::VectorXd &u,
  const Eigen::VectorXd &x_ref, const Eigen::VectorXd &u_ref,
  const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R)
{
Eigen::VectorXd dx = x - x_ref;
Eigen::VectorXd du = u - u_ref;
return (dx.transpose() * Q * dx)(0,0) + (du.transpose() * R * du)(0,0);
}

/**
 * @brief 终端代价：l_f(x) = (x - x_ref)^T Q_f (x - x_ref)
 */
double LQRController::finalCost(const Eigen::VectorXd &x, const Eigen::VectorXd &x_ref,
  const Eigen::MatrixXd &Qf)
{
Eigen::VectorXd dx = x - x_ref;
return (dx.transpose() * Qf * dx)(0,0);
}

/**
 * @brief 计算运行代价关于状态和控制的一阶、二阶导数
 *
 * 因为代价是二次型：
 *   l_x   = 2 Q (x - x_ref)
 *   l_u   = 2 R (u - u_ref)
 *   l_xx  = 2 Q,    l_uu = 2 R,    l_ux = 0
 */
void LQRController::runningCostDerivatives(const Eigen::VectorXd &x, const Eigen::VectorXd &u,
                            const Eigen::VectorXd &x_ref, const Eigen::VectorXd &u_ref,
                            const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
                            Eigen::VectorXd &l_x, Eigen::VectorXd &l_u,
                            Eigen::MatrixXd &l_xx, Eigen::MatrixXd &l_uu,
                            Eigen::MatrixXd &l_ux)
{
    Eigen::VectorXd dx = x - x_ref;
    Eigen::VectorXd du = u - u_ref;
    l_x = 2.0 * Q * dx;
    l_u = 2.0 * R * du;
    l_xx = 2.0 * Q;
    l_uu = 2.0 * R;
    l_ux = Eigen::MatrixXd::Zero(control_dim, state_dim);
}

/**
 * @brief 计算终端代价关于状态的一阶、二阶导数
 */
void LQRController::finalCostDerivatives(const Eigen::VectorXd &x, const Eigen::VectorXd &x_ref,
                          const Eigen::MatrixXd &Qf,
                          Eigen::VectorXd &l_x, Eigen::MatrixXd &l_xx)
{
    Eigen::VectorXd dx = x - x_ref;
    l_x = 2.0 * Qf * dx;
    l_xx = 2.0 * Qf;
}

/**
 * @brief iLQR 算法实现
 *
 * @param x0           初始状态
 * @param u_seq        初始控制序列（长度为 horizon）
 * @param horizon      优化时域步数
 * @param dt           离散时间步长
 * @param x_ref_traj   状态参考轨迹（长度为 horizon+1）
 * @param u_ref_traj   控制参考序列（长度为 horizon）
 * @param Q            运行代价中状态的权重矩阵
 * @param R            运行代价中控制的权重矩阵
 * @param Qf           终端代价权重矩阵
 * @param x_seq_out    优化得到的状态轨迹（输出，长度为 horizon+1）
 * @param u_seq_out    优化得到的控制序列（输出，长度为 horizon）
 */
void LQRController::iLQR(const Eigen::VectorXd &x0, std::vector<Eigen::VectorXd> &u_seq, int horizon, double dt,
          const std::vector<Eigen::VectorXd> &x_ref_traj, const std::vector<Eigen::VectorXd> &u_ref_traj,
          const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &Qf,
          std::vector<Eigen::VectorXd> &x_seq_out, std::vector<Eigen::VectorXd> &u_seq_out)
{
    int max_iter = 100;
    double tol = 1e-6;
    double alpha;
    double cost_total, cost_new;
    int n_x = state_dim;
    int n_u = control_dim;

    // 当前控制序列（初始化为 u_seq 输入）
    std::vector<Eigen::VectorXd> u_seq_current = u_seq;
    std::vector<Eigen::VectorXd> x_seq(horizon+1, Eigen::VectorXd::Zero(n_x));

    for (int iter = 0; iter < max_iter; iter++) {
        // 前向仿真：计算状态轨迹与当前总代价
        x_seq[0] = x0;
        cost_total = 0.0;
        for (int t = 0; t < horizon; t++) {
            x_seq[t+1] = quadrotorDynamics(x_seq[t], u_seq_current[t], dt);
            cost_total += runningCost(x_seq[t], u_seq_current[t],
                                      x_ref_traj[t], u_ref_traj[t],
                                      Q, R);
        }
        cost_total += finalCost(x_seq[horizon], x_ref_traj[horizon], Qf);

        // 向后传播：计算反馈增益和 feedforward 修正项
        std::vector<Eigen::VectorXd> k_seq(horizon, Eigen::VectorXd::Zero(n_u));
        std::vector<Eigen::MatrixXd> K_seq(horizon, Eigen::MatrixXd::Zero(n_u, n_x));

        // 终端代价导数
        Eigen::VectorXd V_x(n_x);
        Eigen::MatrixXd V_xx(n_x, n_x);
        finalCostDerivatives(x_seq[horizon], x_ref_traj[horizon], Qf, V_x, V_xx);

        for (int t = horizon - 1; t >= 0; t--) {
            // 线性化动力学：计算 A, B
            Eigen::MatrixXd A(n_x, n_x), B(n_x, n_u);
            finiteDifferenceDynamics(x_seq[t], u_seq_current[t], dt, A, B);

            // 运行代价导数
            Eigen::VectorXd l_x(n_x), l_u(n_u);
            Eigen::MatrixXd l_xx(n_x, n_x), l_uu(n_u, n_u), l_ux(n_u, n_x);
            runningCostDerivatives(x_seq[t], u_seq_current[t],
                                   x_ref_traj[t], u_ref_traj[t],
                                   Q, R, l_x, l_u, l_xx, l_uu, l_ux);

            // Q函数的局部二次近似
            Eigen::VectorXd Q_x = l_x + A.transpose() * V_x;
            Eigen::VectorXd Q_u = l_u + B.transpose() * V_x;
            Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
            Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
            Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

            // 为确保 Q_uu 正定，加入正则项
            Q_uu += 1e-6 * Eigen::MatrixXd::Identity(n_u, n_u);

            Eigen::MatrixXd inv_Q_uu = Q_uu.inverse();
            Eigen::VectorXd k = -inv_Q_uu * Q_u;
            Eigen::MatrixXd K = -inv_Q_uu * Q_ux;

            k_seq[t] = k;
            K_seq[t] = K;

            // 更新值函数
            V_x = Q_x + K.transpose() * Q_uu * k + K.transpose() * Q_u + Q_ux.transpose() * k;
            V_xx = Q_xx + K.transpose() * Q_uu * K + K.transpose() * Q_ux + Q_ux.transpose() * K;
            V_xx = 0.5 * (V_xx + V_xx.transpose());
        }

        // 前向更新：线搜索
        alpha = 1.0;
        bool found_better = false;
        std::vector<Eigen::VectorXd> x_seq_new(horizon+1, Eigen::VectorXd::Zero(n_x));
        std::vector<Eigen::VectorXd> u_seq_new(horizon, Eigen::VectorXd::Zero(n_u));
        while (alpha > 1e-4) {
            x_seq_new[0] = x0;
            cost_new = 0.0;
            for (int t = 0; t < horizon; t++) {
                // 控制更新公式： u_new = u_current + alpha * k + K * (x_new - x_current)
                u_seq_new[t] = u_seq_current[t] + alpha * k_seq[t] + K_seq[t] * (x_seq_new[t] - x_seq[t]);
                x_seq_new[t+1] = quadrotorDynamics(x_seq_new[t], u_seq_new[t], dt);
                cost_new += runningCost(x_seq_new[t], u_seq_new[t],
                                        x_ref_traj[t], u_ref_traj[t],
                                        Q, R);
            }
            cost_new += finalCost(x_seq_new[horizon], x_ref_traj[horizon], Qf);
            if (cost_new < cost_total) {
                found_better = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!found_better) {
            ROS_WARN("iLQR: Line search failed to improve cost, terminating.");
            break;
        }
        if (std::abs(cost_total - cost_new) < tol) {
            ROS_INFO("iLQR converged at iteration %d", iter);
            u_seq_current = u_seq_new;
            x_seq = x_seq_new;
            break;
        }
        u_seq_current = u_seq_new;
    }

    x_seq_out = x_seq;
    u_seq_out = u_seq_current;
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


  // // 定义 LQR 的权重矩阵 Q 和 R
  Q_ = MatrixXd::Zero(state_dim, state_dim);
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

  R_ = MatrixXd::Identity(control_dim, control_dim); // 控制输入代价
}

VectorXd LQRController::constructStateVector(
  const quadrotor_common::QuadStateEstimate& state) {
  VectorXd x(state_dim);
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
  VectorXd x_ref(state_dim);

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


/**
 * @brief 使用 iLQR 算法实现轨迹跟踪
 *
 * 1. 将当前状态转换为状态向量 x0；
 * 2. 根据参考状态构造参考状态 x_ref 和参考控制 u_eq（参考推力计算）；
 * 3. 构造时域为 horizon（例如 10 步，dt=0.1s）的参考轨迹（假设恒定参考）和初始控制序列；
 * 4. 调用 iLQR 算法迭代优化控制序列；
 * 5. 取优化后序列的第一步控制作为输出，并进行饱和处理。
 */
quadrotor_common::ControlCommand LQRController::run(
  const quadrotor_common::QuadStateEstimate& state_estimate,
  const quadrotor_common::Trajectory& reference_trajectory,
  const LQRControllerParams& config) {
  quadrotor_common::ControlCommand command;
  command.armed = true;
  command.control_mode = quadrotor_common::ControlMode::BODY_RATES;
  if (reference_trajectory.points.empty()) {
    ROS_WARN("Empty trajectory provided. Returning off command.");
    return off();
  }
  const auto& reference_state = reference_trajectory.points.front();
  computeControl(state_estimate, reference_state, &command);
  return command;
}

void LQRController::computeControl(
const quadrotor_common::QuadStateEstimate& state_estimate,
const quadrotor_common::TrajectoryPoint& reference_state,
quadrotor_common::ControlCommand* command)
{
  // 1. 将当前状态转换为状态向量 x0
  Eigen::VectorXd x0 = constructStateVector(state_estimate);
  // 2. 构造参考状态向量 x_ref
  Eigen::VectorXd x_ref = constructReferenceVector(reference_state);
  // 3. 计算参考推力，并构造参考控制 u_eq
  double T_ref = computeReferenceThrust(reference_state);
  Eigen::VectorXd u_eq(control_dim);
  u_eq << T_ref, 0.0, 0.0, 0.0;

  // 4. 定义 iLQR 参数：时域步长 dt 和步数 horizon
  double dt = 0.1;
  int horizon = 15;

  // 5. 构造参考轨迹：假设恒定参考
  std::vector<Eigen::VectorXd> x_ref_traj(horizon+1, x_ref);
  std::vector<Eigen::VectorXd> u_ref_traj(horizon, u_eq);
  // 6. 初始控制序列：全部设为 u_eq
  std::vector<Eigen::VectorXd> u_seq(horizon, u_eq);



  // // 5. 构造状态参考轨迹 x_ref_traj
  // std::vector<Eigen::VectorXd> x_ref_traj;
  // int x_ref_traj_t = 0;
  // for (const auto &traj_point : reference_state.points) {
  //     if (x_ref_traj_t > horizon)
  //         break;
  //     Eigen::VectorXd x_ref = constructReferenceVector(traj_point);
  //     x_ref_traj.push_back(x_ref);
  //     ++x_ref_traj_t;
  // }



  // 3. 构造控制参考序列 u_ref_traj
  // std::vector<Eigen::VectorXd> u_ref_traj;
  // int u_ref_traj_t = 0;
  // for (const auto &traj_point : reference_state.points) {
  //   if (u_ref_traj_t > horizon-9)
  //       break;
  //   double T_ref = computeReferenceThrust(traj_point);
  //   Eigen::VectorXd u_ref(control_dim);
  //   u_ref << T_ref, 0.0, 0.0, 0.0;
  //   u_ref_traj.push_back(u_ref);
  //   ++u_ref_traj_t;
  // }

  // std::vector<Eigen::VectorXd> u_seq = u_ref_traj;

  // 7. 调用 iLQR 算法进行轨迹优化
  std::vector<Eigen::VectorXd> x_seq_out;
  std::vector<Eigen::VectorXd> u_seq_out;
  iLQR(x0, u_seq, horizon, dt, x_ref_traj, u_ref_traj, Q_, R_, Q_, x_seq_out, u_seq_out);

  // 8. 取优化后序列的第一步控制作为输出，并进行饱和处理
  Eigen::VectorXd u = u_seq_out[0];
  saturateInput(u);
  command->collective_thrust = u(0);
  command->bodyrates = u.segment(1, 3);
}

}

