//
// Created by meng on 2020/9/9.
//
#include "mini_snap_trajectory_generator/trajectory_generator.h"
#include <iostream>

using namespace std;
using namespace Eigen;

/*!
 * 计算x的阶乘
 * @param x
 * @return x!
 */
int MiniSnapTrajectoryGeneratorTool::Factorial(int x) {
    int fac = 1;
    for (int i = x; i > 0; i--)
        fac = fac * i;
    return fac;
}

/*!
 * 通过闭式求解QP，得到每段拟合轨迹的多项式系数
 * @param order 导数阶数。例如最小化jerk，则需要求解三次导数，则 d_order=3
 * @param Path 航迹点的空间坐标(3D)
 * @param Vel 航迹点对应的速度(中间点速度是待求的未知量)
 * @param Acc 航迹点对应的加速度(中间点加速度是待求的未知量)
 * @param Time 每段轨迹对应的时间周期
 * @return 轨迹x,y,z三个方向上的多项式系数
 *
 * 返回矩阵(PolyCoeff)的数据格式：每一行是一段轨迹，第一列是x方向上的多项式次数，越左次数越高
 * 第一段轨迹三个方向上的系数 | px_i px_(i-1) px_(i-2) ... px_1 px_0 | y ... | z ... |
 * 第二段轨迹三个方向上的系数 | px_i px_(i-1) px_(i-2) ... px_1 px_0 | y ... | z ... |
 *                          ........
 *
 * 注意：给定起始点和终点的速度加速度，更高阶的导数设置为0
 */
Eigen::MatrixXd MiniSnapTrajectoryGeneratorTool::SolveQPClosedForm(
        int order,
        const Eigen::MatrixXd &Path,
        const Eigen::MatrixXd &Vel,
        const Eigen::MatrixXd &Acc,
        const Eigen::VectorXd &Time) {

    const int p_order = 2 * order - 1;//多项式的最高次数 p^(p_order)t^(p_order) + ...
    const int p_num1d = p_order + 1;//每一段轨迹的变量个数，对于五阶多项式为：p5, p4, ... p0

    const int number_segments = Time.size();
    //每一段都有x,y,z三个方向，每一段多项式的系数的个数有3*p_num1d
    MatrixXd PolyCoeff = MatrixXd::Zero(number_segments, 3 * p_num1d);
    //整条轨迹在ｘ,y,z方向上共多少个未知系数
    const int number_coefficients = p_num1d * number_segments;
    VectorXd Px(number_coefficients), Py(number_coefficients), Pz(number_coefficients);

    const int M_block_rows = order * 2;
    const int M_block_cols = p_num1d;
    //M：转换矩阵，将系数向量转换为方程的微分量
    MatrixXd M = MatrixXd::Zero(number_segments * M_block_rows, number_segments * M_block_cols);
    for (int i = 0; i < number_segments; ++i) {
        int row = i * M_block_rows, col = i * M_block_cols;
        MatrixXd sub_M = MatrixXd::Zero(M_block_rows, M_block_cols);

        for (int j = 0; j < order; ++j) {
            for (int k = 0; k < p_num1d; ++k) {
                if (k < j)
                    continue;

                sub_M(j, p_num1d - 1 - k) = Factorial(k) / Factorial(k - j) * pow(0, k - j);
                sub_M(j + order, p_num1d - 1 - k) = Factorial(k) / Factorial(k - j) * pow(Time(i), k - j);
            }
        }

        M.block(row, col, M_block_rows, M_block_cols) = sub_M;
    }

    //构造选择矩阵C的过程非常复杂，但是只要多花点时间探索一些规律，举几个例子，应该是能写出来的!!
    const int number_valid_variables = (number_segments + 1) * order;
    const int number_fixed_variables = 2 * order + (number_segments - 1);
    //C_T：选择矩阵，用于分离未知量和已知量
    MatrixXd C_T = MatrixXd::Zero(number_coefficients, number_valid_variables);
    for (int i = 0; i < number_coefficients; ++i) {
        if (i < order) {
            C_T(i, i) = 1;
            continue;
        }

        if (i >= number_coefficients - order) {
            const int delta_index = i - (number_coefficients - order);
            C_T(i, number_fixed_variables - order + delta_index) = 1;
            continue;
        }

        if ((i % order == 0) && (i / order % 2 == 1)) {
            const int index = i / (2 * order) + order;
            C_T(i, index) = 1;
            continue;
        }

        if ((i % order == 0) && (i / order % 2 == 0)) {
            const int index = i / (2 * order) + order - 1;
            C_T(i, index) = 1;
            continue;
        }

        if ((i % order != 0) && (i / order % 2 == 1)) {
            const int temp_index_0 = i / (2 * order) * (2 * order) + order;
            const int temp_index_1 = i / (2 * order) * (order - 1) + i - temp_index_0 - 1;
            C_T(i, number_fixed_variables + temp_index_1) = 1;
            continue;
        }

        if ((i % order != 0) && (i / order % 2 == 0)) {
            const int temp_index_0 = (i - order) / (2 * order) * (2 * order) + order;
            const int temp_index_1 = (i - order) / (2 * order) * (order - 1) + (i - order) - temp_index_0 - 1;
            C_T(i, number_fixed_variables + temp_index_1) = 1;
            continue;
        }
    }

    // Q：二项式的系数矩阵
    MatrixXd Q = MatrixXd::Zero(number_coefficients, number_coefficients);
    for (int k = 0; k < number_segments; ++k) {
        MatrixXd sub_Q = MatrixXd::Zero(p_num1d, p_num1d);
        for (int i = 0; i <= p_order; ++i) {
            for (int l = 0; l <= p_order; ++l) {
                if (p_num1d - i <= order || p_num1d - l <= order)
                    continue;

                sub_Q(i, l) = (Factorial(p_order - i) / Factorial(p_order - order - i)) *
                              (Factorial(p_order - l) / Factorial(p_order - order - l)) /
                              (p_order - i + p_order - l - (2 * order - 1)) *
                              pow(Time(k), p_order - i + p_order - l - (2 * order - 1));
            }
        }

        const int row = k * p_num1d;
        Q.block(row, row, p_num1d, p_num1d) = sub_Q;
    }

    MatrixXd R = C_T.transpose() * M.transpose().inverse() * Q * M.inverse() * C_T;

    for (int axis = 0; axis < 3; ++axis) {
        VectorXd d_selected = VectorXd::Zero(number_valid_variables);
        for (int i = 0; i < number_coefficients; ++i) {
            if (i == 0) {
                d_selected(i) = Path(0, axis);
                continue;
            }

            if (i == 1 && order >= 2) {
                d_selected(i) = Vel(0, axis);
                continue;
            }

            if (i == 2 && order >= 3) {
                d_selected(i) = Acc(0, axis);
                continue;
            }

            if (i == number_coefficients - order + 2 && order >= 3) {
                d_selected(number_fixed_variables - order + 2) = Acc(1, axis);
                continue;
            }

            if (i == number_coefficients - order + 1 && order >= 2) {
                d_selected(number_fixed_variables - order + 1) = Vel(1, axis);
                continue;
            }

            if (i == number_coefficients - order) {
                d_selected(number_fixed_variables - order) = Path(number_segments, axis);
                continue;
            }

            if ((i % order == 0) && (i / order % 2 == 0)) {
                const int index = i / (2 * order) + order - 1;
                d_selected(index) = Path(i / (2 * order), axis);
                continue;
            }
        }

        MatrixXd R_PP = R.block(number_fixed_variables, number_fixed_variables,
                                number_valid_variables - number_fixed_variables,
                                number_valid_variables - number_fixed_variables);
        VectorXd d_F = d_selected.head(number_fixed_variables);
        MatrixXd R_FP = R.block(0, number_fixed_variables, number_fixed_variables,
                                number_valid_variables - number_fixed_variables);

        MatrixXd d_optimal = -R_PP.inverse() * R_FP.transpose() * d_F;

        d_selected.tail(number_valid_variables - number_fixed_variables) = d_optimal;
        VectorXd d = C_T * d_selected;

        if (axis == 0)
            Px = M.inverse() * d;

        if (axis == 1)
            Py = M.inverse() * d;

        if (axis == 2)
            Pz = M.inverse() * d;
    }

    for (int i = 0; i < number_segments; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (j == 0) {
                PolyCoeff.block(i, j * p_num1d, 1, p_num1d) =
                        Px.block(p_num1d * i, 0, p_num1d, 1).transpose();
                continue;
            }

            if (j == 1) {
                PolyCoeff.block(i, j * p_num1d, 1, p_num1d) =
                        Py.block(p_num1d * i, 0, p_num1d, 1).transpose();
                continue;
            }

            if (j == 2) {
                PolyCoeff.block(i, j * p_num1d, 1, p_num1d) =
                        Pz.block(p_num1d * i, 0, p_num1d, 1).transpose();
                continue;
            }
        }
    }

    return PolyCoeff;
    /// 该矩阵的实现其数学原理并不难，但是矩阵构造的细节实在是复杂啊
}



MiniSnapTrajectoryGeneratorTool::MiniSnapTrajectoryGeneratorTool(){
    Vel=4.0;//当前机器人能运行的最大速度
    Acc=2.0;//当前机器人能运行的最大加速度
    dev_order= 4;
    min_order=3;
    visualization_traj_width=0.15;

    // waypoint_traj_vis_pub = nh.advertise<visualization_msgs::Marker>("vis_trajectory", 1, true);
    // waypoint_path_vis_pub = nh.advertise<visualization_msgs::Marker>("vis_waypoint_path", 1, true);

    //_poly_numID is the maximum order of polynomial
    poly_coeff_num = 2 * dev_order;

}

/*!
 * 订阅rviz发布的waypoints
 * @param wp
 */
void MiniSnapTrajectoryGeneratorTool::rcvWaypointsCallBack(const nav_msgs::Path &wp) {
    vector<Vector3d> wp_list;
    wp_list.clear();

    for (int k = 0; k < (int) wp.poses.size(); k++) {
        Vector3d pt(wp.poses[k].pose.position.x, wp.poses[k].pose.position.y, wp.poses[k].pose.position.z);
        wp_list.push_back(pt);

        if (wp.poses[k].pose.position.z < 0.0)
            break;
    }

    MatrixXd waypoints(wp_list.size() + 1, 3);
    waypoints.row(0) = start_position;

    for (int k = 0; k < (int) wp_list.size(); k++)
        waypoints.row(k + 1) = wp_list[k];

    TrajGeneration(waypoints);
}

quadrotor_common::Trajectory MiniSnapTrajectoryGeneratorTool::TrajGeneration(const Eigen::MatrixXd &path) {
    
    quadrotor_common::Trajectory full_traj;
    full_traj.timestamp = ros::Time::now();  // 设置下时间戳
    full_traj.trajectory_type = quadrotor_common::Trajectory::TrajectoryType::SNAP; // 示例

    // 用于累计全局的 time_from_start
    double total_time_from_start = 0.0;
    int count = 0;

    MatrixXd vel = MatrixXd::Zero(2, 3);
    MatrixXd acc = MatrixXd::Zero(2, 3);

    vel.row(0) = start_velocity;
    // vel.row(1) = Vector3d(-1, -1, -1);//结束点的速度

    ros::Time start_time = ros::Time::now();
    segment_traj_time = timeAllocation(path);


    poly_coeff = SolveQPClosedForm(dev_order, path, vel, acc, segment_traj_time);
    ros::Duration use_time = (ros::Time::now() - start_time);
    ROS_INFO("\033[1;32m --> Generate trajectory by closed form solution use time: %f (ms)\033[0m",
             use_time.toSec() * 1000);

    // visWayPointPath(path);
    // visWayPointTraj(poly_coeff, segment_traj_time);

    for (int seg_idx = 0; seg_idx < segment_traj_time.size(); seg_idx++) {
        // 每段采样间隔，可根据需求进行修改
        const double dt = 0.01;  
        for (double t_seg = 0.0; t_seg <= segment_traj_time[seg_idx]; t_seg += dt, count += 1) {
            // ============ 1. 获取位置、速度、加速度、jerk等 ============
            Vector3d pos = getDerivativePoly(poly_coeff, seg_idx, t_seg, 0);  // 0阶=位置
            Vector3d vel = getDerivativePoly(poly_coeff, seg_idx, t_seg, 1);  // 1阶=速度
            Vector3d acc = getDerivativePoly(poly_coeff, seg_idx, t_seg, 2);  // 2阶=加速度
            Vector3d jrk = getDerivativePoly(poly_coeff, seg_idx, t_seg, 3);  // 3阶=jerk
            Vector3d snap = getDerivativePoly(poly_coeff, seg_idx, t_seg, 4);


            // ============ 3. 写入 TrajectoryPoint 结构 ============
            quadrotor_common::TrajectoryPoint traj_pt;
            traj_pt.time_from_start = ros::Duration(total_time_from_start + t_seg);
            traj_pt.position = pos;
            traj_pt.orientation = Eigen::Quaterniond::Identity();
            traj_pt.velocity = vel;
            traj_pt.acceleration = acc;
            traj_pt.jerk = jrk;
            traj_pt.snap = snap;

            // push 进 list
            full_traj.points.push_back(traj_pt);
        }

        // 该段结束后，累加总时间
        total_time_from_start += segment_traj_time[seg_idx];
    }
    return full_traj;
}


void MiniSnapTrajectoryGeneratorTool::visWayPointTraj(MatrixXd polyCoeff, VectorXd time) {
    visualization_msgs::Marker _traj_vis;

    _traj_vis.header.stamp = ros::Time::now();
    _traj_vis.header.frame_id = "map";

    _traj_vis.ns = "traj_node/trajectory_waypoints";
    _traj_vis.id = 0;
    _traj_vis.type = visualization_msgs::Marker::POINTS;
    _traj_vis.action = visualization_msgs::Marker::ADD;
    _traj_vis.scale.x = visualization_traj_width;
    _traj_vis.scale.y = visualization_traj_width;
    _traj_vis.scale.z = visualization_traj_width;
    _traj_vis.pose.orientation.x = 0.0;
    _traj_vis.pose.orientation.y = 0.0;
    _traj_vis.pose.orientation.z = 0.0;
    _traj_vis.pose.orientation.w = 1.0;

    _traj_vis.color.a = 1.0;
    _traj_vis.color.r = 1.0;
    _traj_vis.color.g = 0.0;
    _traj_vis.color.b = 0.0;

    double traj_len = 0.0;
    int count = 0;
    Vector3d cur, pre;
    cur.setZero();
    pre.setZero();

    _traj_vis.points.clear();
    Vector3d pos;
    geometry_msgs::Point pt;


    for (int i = 0; i < time.size(); i++) {
        for (double t = 0.0; t < time(i); t += 0.01, count += 1) {
            pos = getPosPoly(polyCoeff, i, t);
            cur(0) = pt.x = pos(0);
            cur(1) = pt.y = pos(1);
            cur(2) = pt.z = pos(2);
            _traj_vis.points.push_back(pt);

            if (count) traj_len += (pre - cur).norm();
            pre = cur;
            ROS_INFO(" pos(0): %f ",
                pos(0));
            
        }
    }

    waypoint_traj_vis_pub.publish(_traj_vis);
}

void MiniSnapTrajectoryGeneratorTool::visWayPointPath(MatrixXd path) {
    visualization_msgs::Marker points, line_list;
    int id = 0;
    points.header.frame_id = line_list.header.frame_id = "map";
    points.header.stamp = line_list.header.stamp = ros::Time::now();
    points.ns = line_list.ns = "wp_path";
    points.action = line_list.action = visualization_msgs::Marker::ADD;
    points.pose.orientation.w = line_list.pose.orientation.w = 1.0;
    points.pose.orientation.x = line_list.pose.orientation.x = 0.0;
    points.pose.orientation.y = line_list.pose.orientation.y = 0.0;
    points.pose.orientation.z = line_list.pose.orientation.z = 0.0;

    points.id = id;
    line_list.id = id;

    points.type = visualization_msgs::Marker::SPHERE_LIST;
    line_list.type = visualization_msgs::Marker::LINE_STRIP;

    points.scale.x = 0.3;
    points.scale.y = 0.3;
    points.scale.z = 0.3;
    points.color.a = 1.0;
    points.color.r = 0.0;
    points.color.g = 0.0;
    points.color.b = 0.0;

    line_list.scale.x = 0.15;
    line_list.scale.y = 0.15;
    line_list.scale.z = 0.15;
    line_list.color.a = 1.0;


    line_list.color.r = 0.0;
    line_list.color.g = 1.0;
    line_list.color.b = 0.0;

    line_list.points.clear();

    for (int i = 0; i < path.rows(); i++) {
        geometry_msgs::Point p;
        p.x = path(i, 0);
        p.y = path(i, 1);
        p.z = path(i, 2);

        points.points.push_back(p);

        if (i < (path.rows() - 1)) {
            geometry_msgs::Point p_line;
            p_line = p;
            line_list.points.push_back(p_line);
            p_line.x = path(i + 1, 0);
            p_line.y = path(i + 1, 1);
            p_line.z = path(i + 1, 2);
            line_list.points.push_back(p_line);
            ROS_INFO("p_line.x %f",p_line.x);
        }
    }

    // waypoint_path_vis_pub.publish(points);
    // waypoint_path_vis_pub.publish(line_list);
}

/*!
 * 求解第k个轨迹段t时刻对应的位置
 * @param polyCoeff 多项式系数矩阵
 * @param k 轨迹段序号
 * @param t 时刻
 * @return [x,y,z]^T
 */
Vector3d MiniSnapTrajectoryGeneratorTool::getPosPoly(MatrixXd polyCoeff, int k, double t) {
    Vector3d ret;

    for (int dim = 0; dim < 3; dim++) {
        VectorXd coeff = (polyCoeff.row(k)).segment(dim * poly_coeff_num, poly_coeff_num);
        VectorXd time = VectorXd::Zero(poly_coeff_num);

        for (int j = 0; j < poly_coeff_num; j++)
            if (j == 0)
                time(j) = 1.0;
            else
                time(j) = pow(t, j);

        double temp_pose = 0.0;
        for (int i = 0; i < time.rows(); ++i) {
            temp_pose = temp_pose + coeff(i) * time(time.rows() - i - 1);
        }
        ret(dim) = temp_pose;
    }

    return ret;
}

/*!
 * 用于轨迹生成过程中，每段轨迹的分配时间的计算
 * @param Path 轨迹的航点
 * @return 每段轨迹应该对应的时间
 */
VectorXd MiniSnapTrajectoryGeneratorTool::timeAllocation(MatrixXd Path) {
    VectorXd times(Path.rows() - 1);

//#define USE_FIXED_TIME
#ifdef USE_FIXED_TIME
    times.setOnes();
#else
    const double MAX_VEL = 1.0;
    const double MAX_ACCEL = 1.0;
    const double t = MAX_VEL / MAX_ACCEL;
    const double dist_threshold_1 = MAX_ACCEL * t * t;

    double segment_t = 0.0;
    for (unsigned int i = 1; i < Path.rows(); ++i) {
        double delta_dist = (Path.row(i) - Path.row(i - 1)).norm();
        // ROS_INFO(" --> delta_dist use time: %f (ms)",
        //     delta_dist);
        if (delta_dist > dist_threshold_1) {
            segment_t = t * 2 + (delta_dist - dist_threshold_1) / MAX_VEL;
            // ROS_INFO("delta_dist > dist_threshold_1 --> segment_t use time: %f (ms)",
            //     segment_t);
        } else {
            segment_t = std::sqrt(delta_dist / MAX_ACCEL);
            // ROS_INFO(" --> segment_t use time: %f (ms)",
            //     segment_t);
        }

        times[i - 1] = segment_t;
        ROS_INFO("times[i - 1] --> segment_t use time: %f (ms)",
            times[i - 1]);
    }
#endif


    return times;
}



/**
 * @brief 返回轨迹在给定段 k、时刻 t、指定阶数 derivativeOrder 的导数 (0=位置，1=速度，2=加速度，3=jerk, 4=snap...)
 * @param polyCoeff 多段轨迹的多项式系数矩阵，形状: [num_segment, 3 * poly_coeff_num]
 * @param k         第 k 段轨迹
 * @param t         时间 (0 <= t <= segment_traj_time[k])
 * @param derivativeOrder 求导的阶数
 * @return 维度为3的向量，对应{x, y, z}
 */
Vector3d MiniSnapTrajectoryGeneratorTool::getDerivativePoly(const MatrixXd &polyCoeff,
    int k,
    double t,
    int derivativeOrder)
{
    // 最终返回的 { x, y, z }
    Vector3d dValue = Vector3d::Zero();

    // 对于当前段 k，polyCoeff 的 row(k) 对应 3 个方向的多项式系数
    //   x 方向：segment(0~poly_coeff_num-1)
    //   y 方向：segment(poly_coeff_num~2*poly_coeff_num-1)
    //   z 方向：segment(2*poly_coeff_num~3*poly_coeff_num-1)

    // 对 3 个维度分别计算
    for (int dim = 0; dim < 3; ++dim) {
        // 取出该段、该维度的系数
        VectorXd coeff_1d = polyCoeff.row(k).segment(dim * poly_coeff_num, poly_coeff_num);
        double temp_val = 0.0;

        // poly_coeff_num - 1 就是多项式的最高阶
        const int n = poly_coeff_num - 1;  
        // 例如对最小化snap来说，p_order=7，对应poly_coeff_num=8 (系数a0..a7)

        // coeff_1d 的索引i从0到n，对应 t^(n-i) 的项
        //    p(t) = coeff[0]*t^n + coeff[1]*t^(n-1) + ... + coeff[n]*t^0
        // 要计算第 derivativeOrder 阶导数时，需要乘上 (n-i)*(n-i-1)*... (n-i-derivativeOrder+1)
        // 并且 t 的指数要变为 (n-i - derivativeOrder)

        for (int i = 0; i <= n; ++i) {
            int power = n - i;  // 该系数对应的原始 t^power
            if (power >= derivativeOrder) {
                // 计算系数前的阶乘因子: power*(power-1)*...*(power-derivativeOrder+1)
                double factor = 1.0;
                for (int r = 0; r < derivativeOrder; r++) {
                    factor *= (power - r);
                }

                temp_val += coeff_1d(i) * factor * std::pow(t, power - derivativeOrder);
            }
        }
        dValue(dim) = temp_val;
    }

    return dValue;
}


// int main(int argc, char **argv) {
//     ros::init(argc, argv, "traj_node");
//     ros::NodeHandle nh("~");

//     nh.param("planning/vel", Vel, 1.0);//当前机器人能运行的最大速度
//     nh.param("planning/acc", Acc, 1.0);//当前机器人能运行的最大加速度
//     nh.param("planning/dev_order", dev_order, 4);
//     nh.param("planning/min_order", min_order, 3);
//     nh.param("vis/vis_traj_width", visualization_traj_width, 0.15);

//     waypoint_traj_vis_pub = nh.advertise<visualization_msgs::Marker>("vis_trajectory", 1, true);
//     waypoint_path_vis_pub = nh.advertise<visualization_msgs::Marker>("vis_waypoint_path", 1, true);

//     //_poly_numID is the maximum order of polynomial
//     poly_coeff_num = 2 * dev_order;

//     // //state of start point
//     start_position(0) = 0;
//     start_position(1) = 0;
//     start_position(2) = 0;

//     start_velocity(0) = 0;
//     start_velocity(1) = 0;
//     start_velocity(2) = 0;
//     vector<Vector3d> wp_list;

//     Eigen::Vector3d manual_pt1(2.0, 2.0, 3.0);
//     Eigen::Vector3d manual_pt2(3.0, 5.0, 6.0);
//     Eigen::Vector3d manual_pt3(1.0, 2.0, 8.0);
//     Eigen::Vector3d manual_pt4(3.0, 1.0, 7.0);
//     wp_list.push_back(manual_pt1);
//     wp_list.push_back(manual_pt2);
//     wp_list.push_back(manual_pt3);
//     wp_list.push_back(manual_pt4);

//     MatrixXd waypoints(wp_list.size() + 1, 3);
//     waypoints.row(0) = start_position;

//     for (int k = 0; k < (int) wp_list.size(); k++)
//         waypoints.row(k + 1) = wp_list[k];

//     TrajGeneration(waypoints);


//     ROS_INFO_STREAM("Path:\n" << waypoints);

//     ros::Rate rate(100);
//     bool status = ros::ok();
//     while (status) {
//         ros::spinOnce();
//         status = ros::ok();
//         rate.sleep();
//     }
//     return 0;
// }