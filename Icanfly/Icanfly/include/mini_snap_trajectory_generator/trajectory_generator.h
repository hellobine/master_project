#ifndef _TRAJECTORY_GENERATOR_H_
#define _TRAJECTORY_GENERATOR_H_

#include <fstream>
#include <cmath>
#include <random>
#include <ros/ros.h>
#include <ros/console.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <quadrotor_common/control_command.h>
#include <quadrotor_common/quad_state_estimate.h>
#include <quadrotor_common/trajectory.h>
#include <quadrotor_common/trajectory_point.h>
#include <Eigen/Eigen>
#include <vector>

using namespace std;
using namespace Eigen;

class MiniSnapTrajectoryGeneratorTool {
private:
    int Factorial(int x);

public:
    MiniSnapTrajectoryGeneratorTool();

    ~MiniSnapTrajectoryGeneratorTool() = default;



    double visualization_traj_width;
    double Vel, Acc;
    int dev_order;//cost函数对应的导数阶数, =3: 最小化jerk =4: 最小化snap
    //当导数的阶数超过一定阶以后，数值发散的很快，此时数值可能不稳定了，
    //主要原因是会超过int的数值精度范围，目前不能超过6
    int min_order;

    ros::Subscriber way_pts_sub;
    ros::Publisher waypoint_traj_vis_pub, waypoint_path_vis_pub;

    int poly_coeff_num;
    MatrixXd poly_coeff;
    VectorXd segment_traj_time = Vector3d::Zero();
    Vector3d start_position = Vector3d::Zero();
    Vector3d start_velocity = Vector3d::Zero();

    void visWayPointTraj(MatrixXd polyCoeff, VectorXd time);

    void visWayPointPath(MatrixXd path);

    Vector3d getPosPoly(MatrixXd polyCoeff, int k, double t);

    VectorXd timeAllocation(MatrixXd Path);

    quadrotor_common::Trajectory TrajGeneration(const Eigen::MatrixXd &path);

    void rcvWaypointsCallBack(const nav_msgs::Path &wp);

    Eigen::MatrixXd SolveQPClosedForm(
        int order,
        const Eigen::MatrixXd &Path,
        const Eigen::MatrixXd &Vel,
        const Eigen::MatrixXd &Acc,
        const Eigen::VectorXd &Time);

    Vector3d getDerivativePoly(const MatrixXd &polyCoeff,
        int k,
        double t,
        int derivativeOrder);
};

#endif // DRONE_CONTROLLER_H