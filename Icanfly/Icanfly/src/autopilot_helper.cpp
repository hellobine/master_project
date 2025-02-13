#include "autopilot_helper/autopilot_helper.h"

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <quadrotor_common/geometry_eigen_conversions.h>
#include <quadrotor_common/math_common.h>
#include <quadrotor_msgs/ControlCommand.h>
#include <quadrotor_msgs/Trajectory.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <std_msgs/Empty.h>

namespace autopilot_helper {

AutoPilotHelper::AutoPilotHelper(const ros::NodeHandle& nh,
                                 const ros::NodeHandle& pnh)
    : autopilot_feedback_(),
      first_autopilot_feedback_received_(false),
      time_last_feedback_received_(),nh_(nh),pnh_(pnh) {
  pose_pub_ =
      nh_.advertise<geometry_msgs::PoseStamped>("autopilot/pose_command", 1);
      
  velocity_pub_ = nh_.advertise<geometry_msgs::TwistStamped>(
      "autopilot/velocity_command", 1);

  reference_state_pub_ = nh_.advertise<quadrotor_msgs::TrajectoryPoint>(
      "autopilot/reference_state", 1);

  trajectory_pub_ =
      nh_.advertise<quadrotor_msgs::Trajectory>("autopilot/trajectory", 1);

  control_command_input_pub_ = nh_.advertise<quadrotor_msgs::ControlCommand>(
      "autopilot/control_command_input", 1);

  start_pub_ = nh_.advertise<std_msgs::Empty>("autopilot/start", 1);

  force_hover_pub_ = nh_.advertise<std_msgs::Empty>("autopilot/force_hover", 1);

  land_pub_ = nh_.advertise<std_msgs::Empty>("autopilot/land", 1);

  off_pub_ = nh_.advertise<std_msgs::Empty>("autopilot/off", 1);

  autopilot_feedback_sub_ =
      nh_.subscribe("/hummingbird/autopilot/feedback", 1,
                    &AutoPilotHelper::autopilotFeedbackCallback, this);

                    
  // 创建 RViz 轨迹发布者
  marker_pub = nh_.advertise<visualization_msgs::Marker>("global_trajectory", 1, true);

  // 1. 设置 Marker 参数
  marker.header.frame_id = "world";  // 请根据实际坐标系修改
  marker.header.stamp = ros::Time::now();
  marker.ns = "trajectory";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::POINTS;  // 显示为点
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = 0.05;  // 点的大小
  marker.scale.y = 0.05;
  marker.scale.z = 0.05;
  marker.color.r = 1.0;  // 红色
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;
  marker.pose.orientation.w = 1.0;
 
}

AutoPilotHelper::~AutoPilotHelper() {}

bool AutoPilotHelper::feedbackAvailable() const {
  if (!first_autopilot_feedback_received_) {
    return false;
  }

  if (feedbackMessageAge() > kFeedbackValidTimeout_) {
    return false;
  }

  return true;
}

double AutoPilotHelper::feedbackMessageAge() const {
  if (!first_autopilot_feedback_received_) {
    // Just return a "very" large number
    return 100.0 * kFeedbackValidTimeout_;
  }

  return (ros::Time::now() - time_last_feedback_received_).toSec();
}

bool AutoPilotHelper::stateEstimateAvailable() const {
  if (getCurrentStateEstimate().coordinate_frame ==
      quadrotor_common::QuadStateEstimate::CoordinateFrame::INVALID) {
    return false;
  }

  return true;
}

bool AutoPilotHelper::waitForAutopilotFeedback(
    const double timeout_s, const double loop_rate_hz) const {
  const ros::Duration timeout(timeout_s);
  ros::Rate loop_rate(loop_rate_hz);
  const ros::Time start_wait = ros::Time::now();
  while (ros::ok() && (ros::Time::now() - start_wait) <= timeout) {
    ros::spinOnce();
    if (feedbackAvailable()) {
      return true;
    }
    loop_rate.sleep();
  }

  return false;
}

bool AutoPilotHelper::waitForSpecificAutopilotState(
    const autopilot::States& state, const double timeout_s,
    const double loop_rate_hz) const {
  const ros::Duration timeout(timeout_s);
  ros::Rate loop_rate(loop_rate_hz);
  const ros::Time start_wait = ros::Time::now();
  while (ros::ok() && (ros::Time::now() - start_wait) <= timeout) {
    ros::spinOnce();
    if (feedbackAvailable() && getCurrentAutopilotState() == state) {
      return true;
    }
    loop_rate.sleep();
  }

  return false;
}

autopilot::States AutoPilotHelper::getCurrentAutopilotState() const {
  switch (autopilot_feedback_.autopilot_state) {
    case autopilot_feedback_.OFF:
      return autopilot::States::OFF;
    case autopilot_feedback_.START:
      return autopilot::States::START;
    case autopilot_feedback_.HOVER:
      return autopilot::States::HOVER;
    case autopilot_feedback_.LAND:
      return autopilot::States::LAND;
    case autopilot_feedback_.EMERGENCY_LAND:
      return autopilot::States::EMERGENCY_LAND;
    case autopilot_feedback_.BREAKING:
      return autopilot::States::BREAKING;
    case autopilot_feedback_.GO_TO_POSE:
      return autopilot::States::GO_TO_POSE;
    case autopilot_feedback_.VELOCITY_CONTROL:
      return autopilot::States::VELOCITY_CONTROL;
    case autopilot_feedback_.REFERENCE_CONTROL:
      return autopilot::States::REFERENCE_CONTROL;
    case autopilot_feedback_.TRAJECTORY_CONTROL:
      return autopilot::States::TRAJECTORY_CONTROL;
    case autopilot_feedback_.COMMAND_FEEDTHROUGH:
      return autopilot::States::COMMAND_FEEDTHROUGH;
    case autopilot_feedback_.RC_MANUAL:
      return autopilot::States::RC_MANUAL;
    default:
      return autopilot::States::OFF;
  }
}

ros::Duration AutoPilotHelper::getCurrentControlCommandDelay() const {
  return autopilot_feedback_.control_command_delay;
}

ros::Duration AutoPilotHelper::getCurrentControlComputationTime() const {
  return autopilot_feedback_.control_computation_time;
}

ros::Duration AutoPilotHelper::getCurrentTrajectoryExecutionLeftDuration()
    const {
  return autopilot_feedback_.trajectory_execution_left_duration;
}

int AutoPilotHelper::getCurrentTrajectoriesLeftInQueue() const {
  return autopilot_feedback_.trajectories_left_in_queue;
}

quadrotor_common::TrajectoryPoint AutoPilotHelper::getCurrentReferenceState()
    const {
  return quadrotor_common::TrajectoryPoint(autopilot_feedback_.reference_state);
}

Eigen::Vector3d AutoPilotHelper::getCurrentReferencePosition() const {
  return quadrotor_common::geometryToEigen(
      autopilot_feedback_.reference_state.pose.position);
}

Eigen::Vector3d AutoPilotHelper::getCurrentReferenceVelocity() const {
  return quadrotor_common::geometryToEigen(
      autopilot_feedback_.reference_state.velocity.linear);
}

Eigen::Quaterniond AutoPilotHelper::getCurrentReferenceOrientation() const {
  return quadrotor_common::geometryToEigen(
      autopilot_feedback_.reference_state.pose.orientation);
}

double AutoPilotHelper::getCurrentReferenceHeading() const {
  return quadrotor_common::quaternionToEulerAnglesZYX(
             quadrotor_common::geometryToEigen(
                 autopilot_feedback_.reference_state.pose.orientation))
      .z();
}

quadrotor_common::QuadStateEstimate AutoPilotHelper::getCurrentStateEstimate()
    const {
  return quadrotor_common::QuadStateEstimate(
      autopilot_feedback_.state_estimate);
}

Eigen::Vector3d AutoPilotHelper::getCurrentPositionEstimate() const {
  return quadrotor_common::geometryToEigen(
      autopilot_feedback_.state_estimate.pose.pose.position);
}

Eigen::Vector3d AutoPilotHelper::getCurrentVelocityEstimate() const {
  return quadrotor_common::geometryToEigen(
      autopilot_feedback_.state_estimate.twist.twist.linear);
}

Eigen::Quaterniond AutoPilotHelper::getCurrentOrientationEstimate() const {
  return quadrotor_common::geometryToEigen(
      autopilot_feedback_.state_estimate.pose.pose.orientation);
}

double AutoPilotHelper::getCurrentHeadingEstimate() const {
  return quadrotor_common::quaternionToEulerAnglesZYX(
             quadrotor_common::geometryToEigen(
                 autopilot_feedback_.state_estimate.pose.pose.orientation))
      .z();
}

Eigen::Vector3d AutoPilotHelper::getCurrentPositionError() const {
  return getCurrentPositionEstimate() - getCurrentReferencePosition();
}

Eigen::Vector3d AutoPilotHelper::getCurrentVelocityError() const {
  return getCurrentVelocityEstimate() - getCurrentReferenceVelocity();
}

Eigen::Quaterniond AutoPilotHelper::getCurrentOrientationError() const {
  return getCurrentReferenceOrientation().inverse() *
         getCurrentOrientationEstimate();
}

double AutoPilotHelper::getCurrentHeadingError() const {
  return quadrotor_common::wrapMinusPiToPi(getCurrentHeadingEstimate() -
                                           getCurrentReferenceHeading());
}

void AutoPilotHelper::sendPoseCommand(const Eigen::Vector3d& position,
                                      const double heading) const {
  geometry_msgs::PoseStamped pose_cmd;
  pose_cmd.pose.position.x = position.x();
  pose_cmd.pose.position.y = position.y();
  pose_cmd.pose.position.z = position.z();
  pose_cmd.pose.orientation =
      quadrotor_common::eigenToGeometry(Eigen::Quaterniond(
          Eigen::AngleAxisd(quadrotor_common::wrapMinusPiToPi(heading),
                            Eigen::Vector3d::UnitZ())));

  pose_pub_.publish(pose_cmd);
}

void AutoPilotHelper::sendVelocityCommand(const Eigen::Vector3d& velocity,
                                          const double heading_rate) const {
  geometry_msgs::TwistStamped vel_cmd;
  vel_cmd.twist.linear.x = velocity.x();
  vel_cmd.twist.linear.y = velocity.y();
  vel_cmd.twist.linear.z = velocity.z();
  vel_cmd.twist.angular.z = heading_rate;

  velocity_pub_.publish(vel_cmd);
}

void AutoPilotHelper::sendReferenceState(
    const quadrotor_common::TrajectoryPoint& trajectory_point) const {
  reference_state_pub_.publish(trajectory_point.toRosMessage());
}

void AutoPilotHelper::sendTrajectory(
    const quadrotor_common::Trajectory& trajectory) const {
  trajectory_pub_.publish(trajectory.toRosMessage());
}

void AutoPilotHelper::sendControlCommandInput(
    const quadrotor_common::ControlCommand& control_command) const {
  control_command_input_pub_.publish(control_command.toRosMessage());
}

void AutoPilotHelper::sendStart() const {
  start_pub_.publish(std_msgs::Empty());
}

void AutoPilotHelper::sendForceHover() const {
  force_hover_pub_.publish(std_msgs::Empty());
}

void AutoPilotHelper::sendLand() const { land_pub_.publish(std_msgs::Empty()); }

void AutoPilotHelper::sendOff() const { off_pub_.publish(std_msgs::Empty()); }

void AutoPilotHelper::autopilotFeedbackCallback(
    const quadrotor_msgs::AutopilotFeedback::ConstPtr& msg) {
  time_last_feedback_received_ = ros::Time::now();
  //  ROS_WARN_THROTTLE(5, "aaaaaaaaa HOVER state...");

  autopilot_feedback_ = *msg;

  if (!first_autopilot_feedback_received_) {
    first_autopilot_feedback_received_ = true;
  }
}

void AutoPilotHelper::addForwardHeading(quadrotor_common::Trajectory* trajectory) {
  auto iterator(trajectory->points.begin());
  auto iterator_prev(trajectory->points.begin());
  iterator_prev = std::prev(iterator_prev);
  auto iterator_next(trajectory->points.begin());
  iterator_next = std::next(iterator_next);
  auto last_element = trajectory->points.end();
  last_element = std::prev(last_element);
  double time_step;

  for (int i = 0; i < trajectory->points.size(); i++) {
    // do orientation first, since bodyrate conversion will depend on it
    Eigen::Vector3d I_eZ_I(0.0, 0.0, 1.0);
    Eigen::Quaterniond quatDes = Eigen::Quaterniond::FromTwoVectors(
        I_eZ_I, iterator->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81));

    double heading = std::atan2(iterator->velocity.y(), iterator->velocity.x());
    // set full orientation and heading to zero
    Eigen::Quaternion<double> q_heading = Eigen::Quaternion<double>(
        Eigen::AngleAxis<double>(heading, Eigen::Vector3d::UnitZ()));
    Eigen::Quaternion<double> q_orientation = quatDes * q_heading;
    iterator->orientation = q_orientation;
    iterator->heading = 0.0;  // heading is now absorbed in orientation
    iterator->heading_rate = 0.0;
    iterator->heading_acceleration = 0.0;

    Eigen::Vector3d thrust_1;
    Eigen::Vector3d thrust_2;
    // catch case of first and last element
    if (i == 0) {
      thrust_1 = iterator->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      time_step =
          (iterator_next->time_from_start - iterator->time_from_start).toSec();
      thrust_2 = iterator_next->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
    } else if (i < trajectory->points.size() - 1) {
      thrust_1 = iterator_prev->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      time_step =
          (iterator_next->time_from_start - iterator_prev->time_from_start)
              .toSec();
      thrust_2 = iterator_next->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
    } else {
      // at the last point, we extrapolate the acceleration
      thrust_1 = iterator_prev->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      thrust_2 = iterator->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81) +
                 time_step / 2.0 * iterator->jerk;
    }

    thrust_1.normalize();
    thrust_2.normalize();

    Eigen::Vector3d crossProd =
        thrust_1.cross(thrust_2);  // direction of omega, in inertial axes
    Eigen::Vector3d angular_rates_wf = Eigen::Vector3d(0, 0, 0);
    if (crossProd.norm() > 0.0) {
      angular_rates_wf = std::acos(thrust_1.dot(thrust_2)) / time_step *
                         crossProd / crossProd.norm();
    }
    // rotate bodyrates to bodyframe
    iterator->bodyrates = q_orientation.inverse() * angular_rates_wf;

    iterator_prev++;
    iterator++;
    iterator_next++;
  }
}


void AutoPilotHelper::generateEightTrajectory(quadrotor_common::Trajectory &traj_msg) {


  // 2. 轨迹参数设置
  double T = 0.1;                // 每段轨迹持续时间（秒）
  int num_loops = 5;            // 总共 5 圈
  int num_points_per_loop = 100; // 每圈 100 个 waypoint
  int total_waypoints = num_loops * num_points_per_loop;
  Vec3 gravity(0, 0, -9.81);       // 重力加速度

  // 3. 生成原始 waypoints（构造一个 8 字形立体轨迹）
  std::vector<Vec3> waypoints;
  waypoints.reserve(total_waypoints);
  for (int i = 0; i < total_waypoints; ++i) {
    double t = (double)i / num_points_per_loop * 2 * M_PI;
    double x = 3 * cos(t) - 3;
    double y = sin(t);
    double z = 3 * sin(2 * t) / 2 + 4;
    waypoints.push_back(Vec3(x, y, z));
  }

  // 4. 初始化轨迹消息
  traj_msg.timestamp = ros::Time::now();
  traj_msg.trajectory_type = quadrotor_common::Trajectory::TrajectoryType::GENERAL;
  traj_msg.points.clear(); 

  // 5. 起始状态：第一个 waypoint 为起点，速度和加速度初始均为 0
  Vec3 pos0 = waypoints[0];
  Vec3 vel0(0, 0, 0);
  Vec3 acc0(0, 0, 0);
  double cumulative_time = 0.0;  // 用于计算每个点的 time_from_start

  // 每段轨迹采样点数量（可调整）
  int num_samples_per_segment = 10;

  // 6. 对每段 waypoint（从 pos0 到下一个 waypoint）生成 mini-jerk 轨迹段，并采样存储中间点
  for (size_t i = 1; i < waypoints.size(); i++) {
    Vec3 posf = waypoints[i];
    Vec3 velf(0, 0, 0);
    Vec3 accf(0, 0, 0);


    RapidTrajectoryGenerator traj_segment(pos0, vel0, acc0, gravity);
    traj_segment.SetGoalPosition(posf);
    traj_segment.SetGoalVelocity(velf);
    traj_segment.SetGoalAcceleration(accf);
    traj_segment.Generate(T);

    // 对该轨迹段按固定时间间隔采样
    for (int j = 0; j <= num_samples_per_segment; j++) {
      double t_sample = T * j / double(num_samples_per_segment);

      Vec3 pos_sample = traj_segment.GetPosition(t_sample);  // 请确保 GetPosition 接口已实现
      Vec3 vel_sample  = traj_segment.GetVelocity(t_sample);

      quadrotor_common::TrajectoryPoint point;
      point.time_from_start = ros::Duration(cumulative_time + t_sample);
      point.position.x() = pos_sample[0];
      point.position.y() = pos_sample[1];
      point.position.z() = pos_sample[2];

      // 赋值速度
      point.velocity.x() = vel_sample[0];
      point.velocity.y() = vel_sample[1];
      point.velocity.z() = vel_sample[2];

      // 添加到 RViz Marker 中显示
      geometry_msgs::Point rviz_point;
      rviz_point.x = pos_sample[0];
      rviz_point.y = pos_sample[1];
      rviz_point.z = pos_sample[2];

      marker.points.push_back(rviz_point);

      // 将轨迹点添加到轨迹消息中
      traj_msg.points.push_back(point);
    }
    // 更新累计时间，并将当前段终点作为下一段起点
    cumulative_time += T;
    pos0 = posf;
    vel0 = velf;
    acc0 = accf;
  }

  trajectory_generation_helper::heading::addForwardHeading(&traj_msg);

  // 发布轨迹到 RViz
  marker_pub.publish(marker);
}

void AutoPilotHelper::generateCircleTrajectory(quadrotor_common::Trajectory &traj_msg) {

  

  int n_loops = 5;
  double exec_loop_rate = 100.0;
  double circle_velocity = 3.0;
  double radius = 4.0;
  Eigen::Vector3d circle_center = Eigen::Vector3d(0,-4,3);


  traj_msg = trajectory_generation_helper::circles::computeHorizontalCircleTrajectory(
  circle_center,radius, circle_velocity, M_PI_2,
  -(-0.5 + 2 * n_loops) * M_PI, exec_loop_rate);  

  trajectory_generation_helper::heading::addForwardHeading(&traj_msg);

  for (const auto &point : traj_msg.points) {
    // 添加到 RViz Marker 中显示
    geometry_msgs::Point rviz_point;
    rviz_point.x = point.position.x();
    rviz_point.y = point.position.y();
    rviz_point.z = point.position.z();
    marker.points.push_back(rviz_point);
  }
  // 发布轨迹到 RViz
  marker_pub.publish(marker);
}


quadrotor_common::Trajectory AutoPilotHelper::generateMultiPointInspectionTrajectory(
  const Eigen::Vector3d& start_point,
  const std::vector<Eigen::Vector3d>& inspection_points,
  double average_speed,
  double sampling_frequency) {

  // ---------------------------
  // 1. 参数设定
  // ---------------------------
  double traj_sampling_freq = sampling_frequency;  // 采样频率
  double max_velocity = average_speed;              // 采用平均速度作为最大速度
  double max_thrust = 20.0;                           // 最大推力
  double max_roll_pitch_rate = 3.0;                   // 最大滚转/俯仰率
  bool minimum_snap = true;                         // 采用 minimum snap 算法
  double end_yaw = 0.0;                             // 统一的航向（可根据需要修改）

  // ---------------------------
  // 2. 构造完整轨迹经过的点序列
  // ---------------------------
  // 构造一个闭合路径：起点 -> 检查点1 -> 检查点2 -> 检查点3 -> 回到起点
  std::vector<Eigen::Vector3d> all_points;
  all_points.push_back(start_point);
  for (const auto &pt : inspection_points) {
    all_points.push_back(pt);
  }
  // 闭合回到起点
  all_points.push_back(start_point);

  // ---------------------------
  // 3. 配置多项式轨迹规划参数（适用于每一段）
  // ---------------------------
  // 设置求解时对高阶导数（加加速度、加加加速度）加大权重
  Eigen::VectorXd minimization_weights(4);
  minimization_weights << 100.0, 100.0, 1.0, 1.0;

  polynomial_trajectories::PolynomialTrajectorySettings trajectory_settings;
  trajectory_settings.minimization_weights = minimization_weights;
  trajectory_settings.polynomial_order = 11;
  trajectory_settings.continuity_order = 5;
  // 如果你需要额外的经过点，可以利用 trajectory_settings.way_points 填充，
  // 但本示例每一段只规划起点和终点，不需要额外设置

  // ---------------------------
  // 4. 生成各段轨迹，并拼接成完整轨迹
  // ---------------------------
  quadrotor_common::Trajectory complete_trajectory;
  bool first_segment = true;
  for (size_t i = 0; i < all_points.size() - 1; i++) {
    // 当前段起点和终点
    Eigen::Vector3d seg_start = all_points[i];
    Eigen::Vector3d seg_end   = all_points[i + 1];

    // 设置初始状态和终止状态（此处速度设置为零；根据需求可修改为期望值）
    quadrotor_common::TrajectoryPoint seg_init, seg_end_point;
    seg_init.position = seg_start;
    seg_init.velocity = Eigen::Vector3d::Zero();
    seg_init.time_from_start = ros::Duration(0.0);

    seg_end_point.position = seg_end;
    seg_end_point.velocity = Eigen::Vector3d::Zero();
    seg_end_point.time_from_start = ros::Duration(0.0);

    // 计算该段的规划时间（距离/平均速度）
    double distance = (seg_end - seg_start).norm();
    double exec_time = distance / average_speed;

    // 将执行时间封装为单段时间向量
    Eigen::VectorXd segment_times = Eigen::VectorXd::Ones(1);
    segment_times[0] = exec_time;

    // 生成当前段轨迹（采用 minimum snap 算法）
    quadrotor_common::Trajectory segment_traj =
      trajectory_generation_helper::polynomials::generateMinimumSnapTrajectory(
          segment_times, seg_init, seg_end_point, trajectory_settings,
          max_velocity, max_thrust, max_roll_pitch_rate, traj_sampling_freq);

    // 拼接轨迹：如果是第一段，直接将所有点加入；否则跳过该段第一个点（与前段重复）
    if (first_segment) {
      for (const auto &pt : segment_traj.points) {
        complete_trajectory.points.push_back(pt);
      }
      first_segment = false;
    } else {
      bool is_first = true;
      for (const auto &pt : segment_traj.points) {
        if (is_first) {
          is_first = false;
          continue;  // 跳过重复的起点
        }
        complete_trajectory.points.push_back(pt);
      }
    }
  }

  // ---------------------------
  // 5. 添加统一朝向（heading）
  // ---------------------------
  trajectory_generation_helper::heading::addConstantHeading(end_yaw, &complete_trajectory);
  for (const auto &point : complete_trajectory.points) {
    // 添加到 RViz Marker 中显示
    geometry_msgs::Point rviz_point;
    rviz_point.x = point.position.x();
    rviz_point.y = point.position.y();
    rviz_point.z = point.position.z();
    marker.points.push_back(rviz_point);
  }
  // 发布轨迹到 RViz
  marker_pub.publish(marker);
  ROS_INFO("Generated multi-point inspection trajectory with %ld points.", complete_trajectory.points.size());
  return complete_trajectory;
}


//using mini-snap
void AutoPilotHelper::generateCurveTrajectory(quadrotor_common::Trajectory &traj_msg) {

  MiniSnapTrajectoryGeneratorTool miniSnapTrajectoryGenerator;

  // //state of start point
  miniSnapTrajectoryGenerator.start_position(0) = 0;
  miniSnapTrajectoryGenerator.start_position(1) = 0;
  miniSnapTrajectoryGenerator.start_position(2) = 0;

  miniSnapTrajectoryGenerator.start_velocity(0) = 0;
  miniSnapTrajectoryGenerator.start_velocity(1) = 0;
  miniSnapTrajectoryGenerator.start_velocity(2) = 0;
  vector<Vector3d> wp_list;

  Eigen::Vector3d manual_pt1(2.0, 2.0, 3.0);
  Eigen::Vector3d manual_pt2(3.0, 5.0, 6.0);
  Eigen::Vector3d manual_pt3(1.0, 2.0, 8.0);
  Eigen::Vector3d manual_pt4(3.0, 1.0, 7.0);
  wp_list.push_back(manual_pt1);
  wp_list.push_back(manual_pt2);
  wp_list.push_back(manual_pt3);
  wp_list.push_back(manual_pt4);

  MatrixXd waypoints(wp_list.size() + 1, 3);
  waypoints.row(0) = miniSnapTrajectoryGenerator.start_position;

  for (int k = 0; k < (int) wp_list.size(); k++)
      waypoints.row(k + 1) = wp_list[k];

  traj_msg = miniSnapTrajectoryGenerator.TrajGeneration(waypoints);
  trajectory_generation_helper::heading::addForwardHeading(&traj_msg);

  for (const auto &point : traj_msg.points) {
    // 添加到 RViz Marker 中显示
    geometry_msgs::Point rviz_point;
    rviz_point.x = point.position.x();
    rviz_point.y = point.position.y();
    rviz_point.z = point.position.z();
    marker.points.push_back(rviz_point);
  }
  // 发布轨迹到 RViz
  marker_pub.publish(marker);
}

}  // namespace autopilot_helper
