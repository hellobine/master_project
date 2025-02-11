#include "trajectory_generation_helper/heading_trajectory_helper.h"

#include <quadrotor_common/math_common.h>

#include <Eigen/Dense>
#include <list>

namespace trajectory_generation_helper {

namespace heading {

void addConstantHeading(const double heading,
                        quadrotor_common::Trajectory* trajectory) {
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

//    double adapted_heading = 0.0;
//    double angle_error = 0.0;
//    int iter_counter = 0;
//    Eigen::Vector3d x_body_world;
//    do {
//      //      std::printf("iter_counter: %d\n", iter_counter);
//      Eigen::Quaternion<double> q_heading_adapt =
//          Eigen::Quaternion<double>(Eigen::AngleAxis<double>(
//              adapted_heading, Eigen::Matrix<double, 3, 1>::UnitZ()));
//      Eigen::Quaterniond q_combined = quatDes * q_heading_adapt;
//      // TODO: set yaw such that the projection of the body-x axis on the
//      // world xy plane aligns with the world x-axis
//      // 1. compute angle between the two axes
//      x_body_world = q_combined * Eigen::Vector3d::UnitX();
//      x_body_world[2] = 0.0;  // project on xy-plane
//      x_body_world.normalize();
//      angle_error = std::acos(x_body_world.dot(Eigen::Vector3d::UnitX()));
//
//      //      std::printf("angle_error: %.5f\n", angle_error);
//      adapted_heading += 0.001;
//      iter_counter++;
//    } while (angle_error > 0.01);
//    std::printf(
//        "body heading of %.2f resulted in %.3f angle error (found solution in "
//        "%d steps)\n",
//        adapted_heading, angle_error, iter_counter);
//    std::printf("body_x_world: %.4f, %.4f, %.4f\n", x_body_world.x(),
//                x_body_world.y(), x_body_world.z());
    // set full orientation and heading to zero
    Eigen::Quaternion<double> q_heading =
        Eigen::Quaternion<double>(Eigen::AngleAxis<double>(
            heading, Eigen::Matrix<double, 3, 1>::UnitZ()));
//    Eigen::Quaternion<double> q_heading =
//        Eigen::Quaternion<double>(Eigen::AngleAxis<double>(
//            adapted_heading, Eigen::Matrix<double, 3, 1>::UnitZ()));
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


void addForwardHeading(quadrotor_common::Trajectory* trajectory) {
  if (trajectory->points.empty()) return;
  auto& points = trajectory->points;
  const size_t num_points = points.size();
  double prev_heading = 0.0;
  const double eps = 1e-6;

  // 初始化迭代器
  auto it = points.begin();
  auto it_prev = points.begin();  // 前一个点迭代器
  auto it_next = points.begin();  // 后一个点迭代器

  for (size_t i = 0; i < num_points; ++i, ++it) {
    auto& current_point = *it;

    // ========== 1. 计算航向角 ==========
    // ----------------------------------
    Eigen::Vector2d velocity_xy(current_point.velocity.x(), current_point.velocity.y());
    double raw_heading;

    // 处理零速度情况
    if (velocity_xy.norm() < 1e-3 && i > 0) {
      raw_heading = prev_heading;  // 沿用上一时刻航向
    } else {
      raw_heading = std::atan2(current_point.velocity.y(), current_point.velocity.x());

      // 航向角连续性处理（角度展开）
      if (i > 0) {
        while (raw_heading - prev_heading > M_PI) raw_heading -= 2 * M_PI;
        while (raw_heading - prev_heading < -M_PI) raw_heading += 2 * M_PI;
      }
      prev_heading = raw_heading;
    }

    // ========== 2. 计算目标姿态 ==========
    // ------------------------------------
    // 计算推力方向对应的姿态
    Eigen::Vector3d thrust_direction = current_point.acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
    Eigen::Quaterniond quatDes = Eigen::Quaterniond::FromTwoVectors(
        Eigen::Vector3d::UnitZ(), thrust_direction);

    // 合成最终姿态（先对齐推力方向，再叠加航向角）
    Eigen::Quaterniond q_heading(Eigen::AngleAxisd(raw_heading, Eigen::Vector3d::UnitZ()));
    current_point.orientation = q_heading * quatDes;
    current_point.heading = 0.0;

    // ========== 3. 计算角速度 ==========
    // ----------------------------------
    if (num_points == 1) {  // 单点轨迹无需计算
      current_point.bodyrates = Eigen::Vector3d::Zero();
      continue;
    }

    // 初始化 thrust_1 和 thrust_2
    Eigen::Vector3d thrust_1, thrust_2;
    double time_step = 0.0;

    // 根据位置选择参考点
    if (i == 0) {  // 第一个点
      it_next = std::next(it);
      thrust_1 = current_point.acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      thrust_2 = it_next->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      time_step = (it_next->time_from_start - current_point.time_from_start).toSec();
    } else if (i < num_points - 1) {  // 中间点
      it_prev = std::prev(it);
      it_next = std::next(it);
      thrust_1 = it_prev->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      thrust_2 = it_next->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      time_step = (it_next->time_from_start - it_prev->time_from_start).toSec();
    } else {  // 最后一个点
      it_prev = std::prev(it);
      thrust_1 = it_prev->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      thrust_2 = current_point.acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      time_step = (current_point.time_from_start - it_prev->time_from_start).toSec();
      // 外推加速度
      thrust_2 += (thrust_2 - thrust_1).normalized() * time_step * 0.5;
    }

    // 归一化推力方向
    thrust_1.normalize();
    thrust_2.normalize();

    // 计算角速度
    Eigen::Vector3d crossProd = thrust_1.cross(thrust_2);
    double dot_product = thrust_1.dot(thrust_2);

    // 手动限制点积范围 [-1, 1]
    if (dot_product > 1.0) dot_product = 1.0;
    else if (dot_product < -1.0) dot_product = -1.0;

    double angle = std::acos(dot_product);
    Eigen::Vector3d angular_rates_wf = Eigen::Vector3d::Zero();

    if (crossProd.norm() > eps && time_step > eps) {
      angular_rates_wf = (angle / time_step) * crossProd.normalized();
    }

    // 转换到机体坐标系
    current_point.bodyrates = current_point.orientation.inverse() * angular_rates_wf;
  }
}

// void addForwardHeading(quadrotor_common::Trajectory* trajectory) {
//   auto iterator(trajectory->points.begin());
//   auto iterator_prev(trajectory->points.begin());
//   iterator_prev = std::prev(iterator_prev);
//   auto iterator_next(trajectory->points.begin());
//   iterator_next = std::next(iterator_next);
//   auto last_element = trajectory->points.end();
//   last_element = std::prev(last_element);
//   double time_step;

//   for (int i = 0; i < trajectory->points.size(); i++) {
//     // do orientation first, since bodyrate conversion will depend on it
//     Eigen::Vector3d I_eZ_I(0.0, 0.0, 1.0);
//     Eigen::Quaterniond quatDes = Eigen::Quaterniond::FromTwoVectors(
//         I_eZ_I, iterator->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81));

//     double heading = std::atan2(iterator->velocity.y(), iterator->velocity.x());
//     // set full orientation and heading to zero
//     Eigen::Quaternion<double> q_heading = Eigen::Quaternion<double>(
//         Eigen::AngleAxis<double>(heading, Eigen::Vector3d::UnitZ()));
//     Eigen::Quaternion<double> q_orientation = quatDes * q_heading;
//     iterator->orientation = q_orientation;
//     iterator->heading = 0.0;  // heading is now absorbed in orientation
//     iterator->heading_rate = 0.0;
//     iterator->heading_acceleration = 0.0;

//     Eigen::Vector3d thrust_1;
//     Eigen::Vector3d thrust_2;
//     // catch case of first and last element
//     if (i == 0) {
//       thrust_1 = iterator->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
//       time_step =
//           (iterator_next->time_from_start - iterator->time_from_start).toSec();
//       thrust_2 = iterator_next->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
//     } else if (i < trajectory->points.size() - 1) {
//       thrust_1 = iterator_prev->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
//       time_step =
//           (iterator_next->time_from_start - iterator_prev->time_from_start)
//               .toSec();
//       thrust_2 = iterator_next->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
//     } else {
//       // at the last point, we extrapolate the acceleration
//       thrust_1 = iterator_prev->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
//       thrust_2 = iterator->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81) +
//                  time_step / 2.0 * iterator->jerk;
//     }

//     thrust_1.normalize();
//     thrust_2.normalize();

//     Eigen::Vector3d crossProd =
//         thrust_1.cross(thrust_2);  // direction of omega, in inertial axes
//     Eigen::Vector3d angular_rates_wf = Eigen::Vector3d(0, 0, 0);
//     if (crossProd.norm() > 0.0) {
//       angular_rates_wf = std::acos(thrust_1.dot(thrust_2)) / time_step *
//                          crossProd / crossProd.norm();
//     }
//     // rotate bodyrates to bodyframe
//     iterator->bodyrates = q_orientation.inverse() * angular_rates_wf;

//     iterator_prev++;
//     iterator++;
//     iterator_next++;
//   }
// }



void addConstantHeadingRate(const double initial_heading,
                            const double final_heading,
                            quadrotor_common::Trajectory* trajectory) {
  if (trajectory->points.size() < 2) {
    return;
  }
  const double delta_angle =
      final_heading -
      initial_heading;  // quadrotor_common::wrapAngleDifference(initial_heading,
                        // final_heading);
  const double trajectory_duration =
      (trajectory->points.back().time_from_start -
       trajectory->points.front().time_from_start)
          .toSec();
  const double heading_rate = delta_angle / trajectory_duration;
  const double delta_heading = delta_angle / trajectory->points.size();

  double heading = initial_heading;
  std::list<quadrotor_common::TrajectoryPoint>::iterator it;
  for (auto& point : trajectory->points) {
    // do orientation first, since bodyrate conversion will depend on it
    Eigen::Vector3d I_eZ_I(0.0, 0.0, 1.0);
    Eigen::Quaterniond quatDes = Eigen::Quaterniond::FromTwoVectors(
        I_eZ_I, point.acceleration + Eigen::Vector3d(0.0, 0.0, 9.81));

    // set full orientation and heading to zero
    Eigen::Quaternion<double> q_heading =
        Eigen::Quaternion<double>(Eigen::AngleAxis<double>(
            heading, Eigen::Matrix<double, 3, 1>::UnitZ()));
    Eigen::Quaternion<double> q_orientation = quatDes * q_heading;
    point.orientation = q_orientation;
    point.heading = 0.0;  // heading is now absorbed in orientation
    point.heading_rate = 0.0;
    point.heading_acceleration = 0.0;

    heading += delta_heading;

    // since we know the full orientation at this point, we can compute the
    // feedforward bodyrates
    double time_step = 0.02;
    Eigen::Vector3d acc1 = point.acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
    Eigen::Vector3d acc2 =
        point.acceleration + Eigen::Vector3d(0.0, 0.0, 9.81) +
        time_step *
            point.jerk;  // should be acceleration at next trajectory point

    acc1.normalize();
    acc2.normalize();

    Eigen::Vector3d crossProd =
        acc1.cross(acc2);  // direction of omega, in inertial axes
    Eigen::Vector3d bodyrates_wf = Eigen::Vector3d(0, 0, 0);
    if (crossProd.norm() > 0.0) {
      bodyrates_wf =
          std::acos(acc1.dot(acc2)) / time_step * crossProd / crossProd.norm();
    }
    // rotate angular rates to bodyframe
    point.bodyrates = q_orientation.inverse() * bodyrates_wf;
  }
}

}  // namespace heading

}  // namespace trajectory_generation_helper
