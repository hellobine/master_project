#include "trajectory_generation_helper/polynomial_trajectory_helper.h"

#include <polynomial_trajectories/constrained_polynomial_trajectories.h>
#include <polynomial_trajectories/minimum_snap_trajectories.h>
#include <polynomial_trajectories/polynomial_trajectories_common.h>

namespace trajectory_generation_helper {

namespace polynomials {

// Constrained Polynomials
quadrotor_common::Trajectory computeTimeOptimalTrajectory(
    const quadrotor_common::TrajectoryPoint& s0,
    const quadrotor_common::TrajectoryPoint& s1, const int order_of_continuity,
    const double max_velocity, const double max_normalized_thrust,
    const double max_roll_pitch_rate, const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::constrained_polynomial_trajectories::
          computeTimeOptimalTrajectory(s0, s1, order_of_continuity,
                                       max_velocity, max_normalized_thrust,
                                       max_roll_pitch_rate);

  return samplePolynomial(polynomial, sampling_frequency);
}

quadrotor_common::Trajectory computeFixedTimeTrajectory(
    const quadrotor_common::TrajectoryPoint& s0,
    const quadrotor_common::TrajectoryPoint& s1, const int order_of_continuity,
    const double execution_time, const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::constrained_polynomial_trajectories::
          computeFixedTimeTrajectory(s0, s1, order_of_continuity,
                                     execution_time);

  return samplePolynomial(polynomial, sampling_frequency);
}

// Minimum Snap Style Polynomials
quadrotor_common::Trajectory generateMinimumSnapTrajectory(
    const Eigen::VectorXd& segment_times,
    const quadrotor_common::TrajectoryPoint& start_state,
    const quadrotor_common::TrajectoryPoint& end_state,
    const polynomial_trajectories::PolynomialTrajectorySettings&
        trajectory_settings,
    const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::minimum_snap_trajectories::
          generateMinimumSnapTrajectory(segment_times, start_state, end_state,
                                        trajectory_settings);

  return samplePolynomial(polynomial, sampling_frequency);
}

quadrotor_common::Trajectory generateMinimumSnapTrajectory(
    const Eigen::VectorXd& initial_segment_times,
    const quadrotor_common::TrajectoryPoint& start_state,
    const quadrotor_common::TrajectoryPoint& end_state,
    const polynomial_trajectories::PolynomialTrajectorySettings&
        trajectory_settings,
    const double max_velocity, const double max_normalized_thrust,
    const double max_roll_pitch_rate, const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::minimum_snap_trajectories::
          generateMinimumSnapTrajectory(initial_segment_times, start_state,
                                        end_state, trajectory_settings,
                                        max_velocity, max_normalized_thrust,
                                        max_roll_pitch_rate);

  return samplePolynomial(polynomial, sampling_frequency);
}

// quadrotor_common::Trajectory generateFigureEightTrajectory(
//     const Eigen::VectorXd& initial_segment_times,
//     const polynomial_trajectories::PolynomialTrajectorySettings& trajectory_settings,
//     const double max_velocity, const double max_normalized_thrust,
//     const double max_roll_pitch_rate, const double sampling_frequency,
//     const Eigen::Vector3d& center, const double radius_x, const double radius_z) {

//   // 定义 8 字轨迹的两个关键点
//   quadrotor_common::TrajectoryPoint start_state, mid_state, end_state;

//   // 设定时间
//   Eigen::VectorXd half_segment_times = initial_segment_times / 2.0;  // 分两段

//   // **第一段：左上 -> 右下**
//   start_state.position = center + Eigen::Vector3d(-radius_x, 0, radius_z);
//   mid_state.position = center + Eigen::Vector3d(radius_x, 0, -radius_z);
  
//   // 生成第一段轨迹
//   polynomial_trajectories::PolynomialTrajectory poly1 =
//       polynomial_trajectories::minimum_snap_trajectories::
//           generateMinimumSnapTrajectory(half_segment_times, start_state, mid_state,
//                                         trajectory_settings, max_velocity, max_normalized_thrust,
//                                         max_roll_pitch_rate);
  
//   quadrotor_common::Trajectory traj1 = samplePolynomial(poly1, sampling_frequency);

//   // **第二段：右下 -> 左上**
//   end_state.position = start_state.position;  // 回到起点

//   polynomial_trajectories::PolynomialTrajectory poly2 =
//       polynomial_trajectories::minimum_snap_trajectories::
//           generateMinimumSnapTrajectory(half_segment_times, mid_state, end_state,
//                                         trajectory_settings, max_velocity, max_normalized_thrust,
//                                         max_roll_pitch_rate);
  
//   quadrotor_common::Trajectory traj2 = samplePolynomial(poly2, sampling_frequency);

//   // **合并轨迹**
//   quadrotor_common::Trajectory full_trajectory;
//   full_trajectory.points.insert(full_trajectory.points.end(), traj1.points.begin(), traj1.points.end());
//   full_trajectory.points.insert(full_trajectory.points.end(), traj2.points.begin(), traj2.points.end());

//     // **动态更新 heading**
//     if (full_trajectory.points.size() > 1) {
//     auto it = full_trajectory.points.begin();
//     auto next_it = std::next(it);  // 获取下一个元素的迭代器

//     while (next_it != full_trajectory.points.end()) {
//         Eigen::Vector3d dir = next_it->position - it->position;
//         if (dir.norm() > 1e-6) {  // 避免 0 向量导致的 atan2(0,0) 错误
//         double yaw = atan2(dir.y(), dir.x());
//         it->orientation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
//         }
//         ++it;
//         ++next_it;
//     }

//     // **最后一个点的 yaw 角设为倒数第二个点的 yaw**
//     if (full_trajectory.points.size() > 1) {
//         full_trajectory.points.back().orientation = std::prev(full_trajectory.points.end())->orientation;
//     }
//     }

//   full_trajectory.trajectory_type = quadrotor_common::Trajectory::TrajectoryType::GENERAL;

//   return full_trajectory;
// }





quadrotor_common::Trajectory generateMinimumSnapTrajectoryWithSegmentRefinement(
    const Eigen::VectorXd& initial_segment_times,
    const quadrotor_common::TrajectoryPoint& start_state,
    const quadrotor_common::TrajectoryPoint& end_state,
    const polynomial_trajectories::PolynomialTrajectorySettings&
        trajectory_settings,
    const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::minimum_snap_trajectories::
          generateMinimumSnapTrajectoryWithSegmentRefinement(
              initial_segment_times, start_state, end_state,
              trajectory_settings);

  return samplePolynomial(polynomial, sampling_frequency);
}

quadrotor_common::Trajectory generateMinimumSnapTrajectoryWithSegmentRefinement(
    const Eigen::VectorXd& initial_segment_times,
    const quadrotor_common::TrajectoryPoint& start_state,
    const quadrotor_common::TrajectoryPoint& end_state,
    const polynomial_trajectories::PolynomialTrajectorySettings&
        trajectory_settings,
    const double max_velocity, const double max_normalized_thrust,
    const double max_roll_pitch_rate, const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::minimum_snap_trajectories::
          generateMinimumSnapTrajectoryWithSegmentRefinement(
              initial_segment_times, start_state, end_state,
              trajectory_settings, max_velocity, max_normalized_thrust,
              max_roll_pitch_rate);

  return samplePolynomial(polynomial, sampling_frequency);
}

quadrotor_common::Trajectory generateMinimumSnapRingTrajectory(
    const Eigen::VectorXd& segment_times,
    const polynomial_trajectories::PolynomialTrajectorySettings&
        trajectory_settings,
    const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::minimum_snap_trajectories::
          generateMinimumSnapRingTrajectory(segment_times, trajectory_settings);

  return samplePolynomial(polynomial, sampling_frequency);
}

quadrotor_common::Trajectory generateMinimumSnapRingTrajectory(
    const Eigen::VectorXd& initial_segment_times,
    const polynomial_trajectories::PolynomialTrajectorySettings&
        trajectory_settings,
    const double max_velocity, const double max_normalized_thrust,
    const double max_roll_pitch_rate, const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::minimum_snap_trajectories::
          generateMinimumSnapRingTrajectory(
              initial_segment_times, trajectory_settings, max_velocity,
              max_normalized_thrust, max_roll_pitch_rate);

  return samplePolynomial(polynomial, sampling_frequency);
}

quadrotor_common::Trajectory
generateMinimumSnapRingTrajectoryWithSegmentRefinement(
    const Eigen::VectorXd& initial_segment_times,
    const polynomial_trajectories::PolynomialTrajectorySettings&
        trajectory_settings,
    const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::minimum_snap_trajectories::
          generateMinimumSnapRingTrajectoryWithSegmentRefinement(
              initial_segment_times, trajectory_settings);

  return samplePolynomial(polynomial, sampling_frequency);
}

quadrotor_common::Trajectory
generateMinimumSnapRingTrajectoryWithSegmentRefinement(
    const Eigen::VectorXd& initial_segment_times,
    const polynomial_trajectories::PolynomialTrajectorySettings&
        trajectory_settings,
    const double max_velocity, const double max_normalized_thrust,
    const double max_roll_pitch_rate, const double sampling_frequency) {
  polynomial_trajectories::PolynomialTrajectory polynomial =
      polynomial_trajectories::minimum_snap_trajectories::
          generateMinimumSnapRingTrajectoryWithSegmentRefinement(
              initial_segment_times, trajectory_settings, max_velocity,
              max_normalized_thrust, max_roll_pitch_rate);

  return samplePolynomial(polynomial, sampling_frequency);
}

// Sampling function
quadrotor_common::Trajectory samplePolynomial(
    const polynomial_trajectories::PolynomialTrajectory& polynomial,
    const double sampling_frequency) {
  if (polynomial.trajectory_type ==
      polynomial_trajectories::TrajectoryType::UNDEFINED) {
    return quadrotor_common::Trajectory();
  }

  quadrotor_common::Trajectory trajectory;

  trajectory.points.push_back(polynomial.start_state);

  const ros::Duration dt(1.0 / sampling_frequency);
  ros::Duration time_from_start = polynomial.start_state.time_from_start + dt;

  while (time_from_start < polynomial.T) {
    trajectory.points.push_back(polynomial_trajectories::getPointFromTrajectory(
        polynomial, time_from_start));
    time_from_start += dt;
  }

  trajectory.points.push_back(polynomial.end_state);

  trajectory.trajectory_type =
      quadrotor_common::Trajectory::TrajectoryType::GENERAL;

  return trajectory;
}

}  // namespace polynomials

}  // namespace trajectory_generation_helper
