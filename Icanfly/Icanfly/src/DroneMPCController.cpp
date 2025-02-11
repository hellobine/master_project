#include "mpc_controller/DroneMPCController.h"
// #include "acado_auxiliary_functions.h"

#define RADIAN M_PI / 180.0

int n_loops = 1;
double exec_loop_rate = 100.0;
double circle_velocity = 3.0;
double radius = 3.0;
Eigen::Vector3d circle_center = Eigen::Vector3d(0,0,2);

using namespace casadi;
using namespace mpc_controller;
// 转换 Eigen::MatrixXd 到 CasADi::DM
casadi::DM eigenToDM(const Eigen::MatrixXd& eigen_mat) {
    casadi::DM casadi_mat(eigen_mat.rows(), eigen_mat.cols());
    for (int i = 0; i < eigen_mat.rows(); ++i) {
        for (int j = 0; j < eigen_mat.cols(); ++j) {
            casadi_mat(i, j) = eigen_mat(i, j);
        }
    }
    return casadi_mat;
}


MPC::MPC(double T, double dt) : dt_(dt), N_(int(T/dt)) {
    Q_goal_ = Eigen::MatrixXd::Zero(10, 10);
    Q_goal_.diagonal() << 100, 100, 100, 10, 10, 10, 10, 10, 10, 10;
    Q_pen_ = Eigen::MatrixXd::Zero(10, 10);
    Q_pen_.diagonal() << 0, 100, 100, 10, 10, 10, 10, 0, 10, 10;
    Q_u_ = Eigen::MatrixXd::Identity(4, 4) * 0.1;
    
    Q_goal_mx_ = eigenToDM(Q_goal_);
    Q_pen_mx_ = eigenToDM(Q_pen_);
    Q_u_mx_ = eigenToDM(Q_u_);
    
    initDynamics();
}

void MPC::initDynamics() {
    // State variables
    MX px = MX::sym("px"), py = MX::sym("py"), pz = MX::sym("pz");
    MX qw = MX::sym("qw"), qx = MX::sym("qx"), qy = MX::sym("qy"), qz = MX::sym("qz");
    MX vx = MX::sym("vx"), vy = MX::sym("vy"), vz = MX::sym("vz");
    MX x = MX::vertcat({px, py, pz, qw, qx, qy, qz, vx, vy, vz});

    // Control variables
    MX thrust = MX::sym("thrust"), wx = MX::sym("wx"), wy = MX::sym("wy"), wz = MX::sym("wz");
    MX u = MX::vertcat({thrust, wx, wy, wz});

    // Dynamics
    MX x_dot = MX::vertcat({
        vx,
        vy,
        vz,
        0.5*(-wx*qx - wy*qy - wz*qz),
        0.5*(wx*qw + wz*qy - wy*qz),
        0.5*(wy*qw - wz*qx + wx*qz),
        0.5*(wz*qw + wy*qx - wx*qy),
        2*(qw*qy + qx*qz)*thrust,
        2*(qy*qz - qw*qx)*thrust,
        (qw*qw - qx*qx - qy*qy + qz*qz)*thrust - gz_
    });
    
    dynamics_ = Function("f", {x, u}, {x_dot});
}

Eigen::Vector4d MPC::solve(const Eigen::VectorXd& current_state,
                          const Eigen::MatrixXd& ref_trajectory) {
    Opti opti;
    MX X = opti.variable(state_dim_, N_+1);
    MX U = opti.variable(ctrl_dim_, N_);
    
    // Initial state constraint
    opti.subject_to(X(Slice(), 0) == eigenToDM(current_state));
    
    // Dynamics constraints
    for(int k=0; k<N_; ++k){
        MX x_next = rk4Integrator(X(Slice(),k), U(Slice(),k));
        opti.subject_to(X(Slice(),k+1) == x_next);
    }
    
    // Cost function
    MX cost = 0;
    // N_=10;
    for(int k=0; k<N_; ++k){
        MX state_error = X(Slice(),k) - eigenToDM(ref_trajectory.col(k));
        MX ctrl_error = U(Slice(),k) - DM(std::vector<double>{gz_, 0, 0, 0});
        
        if(k == N_-1)
            cost += mtimes(mtimes(state_error.T(), Q_goal_mx_), state_error);
        else
            cost += mtimes(mtimes(state_error.T(), Q_pen_mx_), state_error);
            
        cost += mtimes(mtimes(ctrl_error.T(), Q_u_mx_), ctrl_error);
        // ROS_INFO_STREAM("MPC Cost: " << cost);  // **打印目标函数**

    }

    // ROS_INFO_STREAM("MPC Cost: " << cost);  // **打印目标函数**

    
    // Control constraints
    opti.subject_to(opti.bounded(thrust_min_, U(0, Slice()), thrust_max_));
    opti.subject_to(opti.bounded(-w_max_xy_, U(1, Slice()), w_max_xy_));
    opti.subject_to(opti.bounded(-w_max_xy_, U(2, Slice()), w_max_xy_));
    opti.subject_to(opti.bounded(-w_max_yaw_, U(3, Slice()), w_max_yaw_));
    
    opti.minimize(cost);
    opti.solver("ipopt");
    
    // Solve NLP
    try {
        casadi::Dict opts;
        opts["print_time"] = false;
        opti.solver("ipopt", opts);
        auto sol = opti.solve();
        DM u_opt = sol.value(U(Slice(),0));
        return Eigen::Vector4d(u_opt(0).scalar(), 
                             u_opt(1).scalar(),
                             u_opt(2).scalar(),
                             u_opt(3).scalar());
    } catch(...) {
        return Eigen::Vector4d(0.68 * 9.81, 0, 0, 0);
    }
}

casadi::MX MPC::rk4Integrator(const casadi::MX& x, const casadi::MX& u) {
    MX k1 = dt_ * dynamics_(MXVector{x, u})[0];
    MX k2 = dt_ * dynamics_(MXVector{x + 0.5*k1, u})[0];
    MX k3 = dt_ * dynamics_(MXVector{x + 0.5*k2, u})[0];
    MX k4 = dt_ * dynamics_(MXVector{x + k3, u})[0];
    return x + (k1 + 2*k2 + 2*k3 + k4)/6;
}

// DroneMpcController implementation
DroneMpcController::DroneMpcController(ros::NodeHandle& nh, ros::NodeHandle& pnh) 
    : mpc_(new MPC(T_, dt_)) {
    control_pub_ = nh.advertise<mav_msgs::RollPitchYawrateThrust>(
        rotors_control::kDefaultCommandRollPitchYawrateThrustTopic, 10);
    odometry_sub_ = nh.subscribe("/hummingbird/odometry_sensor1/odometry", 1,
                                &DroneMpcController::odometryCallback, this);
    marker_pub_ = nh.advertise<visualization_msgs::Marker>("visualization_marker1", 10);
    target_point_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/hummingbird/target_point", 10);
    timer_ = nh.createTimer(ros::Duration(0.1), &DroneMpcController::publishTrajectoryMarkers, this);

    // rpg_mpc::MpcController<double> mpc_controller_(nh, pnh, "/rpg_mpc/trajectory_predicted");

    mpc_params.loadParameters(pnh);

}

void DroneMpcController::publishTrajectoryMarkers(const ros::TimerEvent& event) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "computed_trajectory";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::POINTS;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    for (const auto& point : all_points) {
        geometry_msgs::Point p;
        p.x = point.position.x();
        p.y = point.position.y();
        p.z = point.position.z();
        marker.points.push_back(p);
    }

    marker_pub_.publish(marker);
    // ROS_INFO_STREAM("[ I am pub trajectory!!!!!!!!!!!!!!!!!!!!!!! ]");
}


void DroneMpcController::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
    received_state_est_ = *msg;
    rotors_control::eigenOdometryFromMsg(msg, &odometry_);
}

Eigen::VectorXd DroneMpcController::odomToState(const rotors_control::EigenOdometry& odom) {
    Eigen::VectorXd state(10);
    state << odometry_.position.x(),
            odometry_.position.y(),
            odometry_.position.z(),
            odometry_.orientation.w(),
            odometry_.orientation.x(),
            odometry_.orientation.y(),
            odometry_.orientation.z(),
            odometry_.velocity.x(),
            odometry_.velocity.y(),
            odometry_.velocity.z();
    return state;
}

void DroneMpcController::computeManeuver() {
     all_points.clear();  // 清空历史轨迹点

    // 1. ----------------------------------------------
    // Trajectory circle_trajectory =
    // trajectory_generation_helper::circles::computeVerticalCircleTrajectory(
    //       circle_center, 2 ,radius, circle_velocity, M_PI_2,
    //       -(-0.5+2 * n_loops) * M_PI, exec_loop_rate);

    // 2. ----------------------------------------------
    // Trajectory circle_trajectory = trajectory_generation_helper::circles::computeHorizontalCircleTrajectory(
    //     circle_center,radius, circle_velocity, M_PI_2,
    //     -(-0.5 + 2 * n_loops) * M_PI, exec_loop_rate);        
    // trajectory_generation_helper::heading::addForwardHeading(&circle_trajectory);

    // 3. -----------------------------------------------
    const double max_vel = 2.0;
    const double max_thrust = 15.0;
    const double max_roll_pitch_rate = 2;
    const Eigen::Vector3d position_cmd = Eigen::Vector3d(0.0, 0.0, 0.0);
    quadrotor_common::TrajectoryPoint start_state;
    start_state.position = position_cmd;
    start_state.heading = 0.0;
    quadrotor_common::TrajectoryPoint end_state;
    end_state.position = Eigen::Vector3d(23.5, 25.7, 3.2);
    end_state.heading = M_PI;

    quadrotor_common::Trajectory circle_trajectory =
    trajectory_generation_helper::polynomials::computeTimeOptimalTrajectory(
        start_state, end_state, 5, max_vel, max_thrust, max_roll_pitch_rate,
        50);

    trajectory_generation_helper::heading::addConstantHeadingRate(
    start_state.heading, end_state.heading, &circle_trajectory);
    //--------------------------------------------------
    
    
    std::vector<quadrotor_common::TrajectoryPoint> sampled_trajectory;
    for (auto point : circle_trajectory.points) {
        all_points.push_back(point);
    }
}

void DroneMpcController::run() {
    computeManeuver();
    
    ROS_INFO_STREAM("[ all_points.size: " << all_points.size() << " ]");

    if (all_points.empty()) return;
     ros::Time start_time = ros::Time::now();
    size_t current_point_idx = 0;

    ros::Rate rate(10);
    while (ros::ok()) {
        current_point_idx+=1;

        quadrotor_common::Trajectory reference_trajectory = quadrotor_common::Trajectory();
        reference_trajectory.trajectory_type =
                quadrotor_common::Trajectory::TrajectoryType::GENERAL;

        // Eigen::VectorXd current_state = odomToState(odometry_);


        ros::Time current_time = ros::Time::now();
        ros::Duration elapsed_time = current_time - start_time;

        // **使用二分查找找到最近的轨迹点**
        size_t low = 0, high = all_points.size() - 1;
        while (low < high) {
            size_t mid = (low + high) / 2;
            if (all_points[mid].time_from_start < elapsed_time) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        current_point_idx = low;

        for(int i=current_point_idx; i<current_point_idx + 10; i++){
            reference_trajectory.points.push_back(all_points[i]);
        }

        
        // 2️⃣ **打印 current_state**
        // ROS_INFO_STREAM("Current State: \n" << current_state.transpose());
        quadrotor_common::QuadStateEstimate state_estimate;
        {
            // std::lock_guard<std::mutex> guard(odom_mtx_);
            // received_state_est_ = *msg;
            if (true) {
            received_state_est_.transformVelocityToWorldFrame();
            }
            state_estimate = received_state_est_;
        }


         // 运行 MPC 控制器
        quadrotor_common::ControlCommand command = mpc_controller_.run(state_estimate, reference_trajectory, mpc_params);

        //    // 打印输出的控制指令
        ROS_INFO("Thrust: %f, Body Rates: [%f, %f, %f]",
             command.collective_thrust,
             command.bodyrates.x(), command.bodyrates.y(), command.bodyrates.z());

        mav_msgs::RollPitchYawrateThrust msg;
        msg.thrust.z = command.collective_thrust;
        msg.roll = command.bodyrates.x();
        msg.pitch = command.bodyrates.y();
        msg.yaw_rate = command.bodyrates.z();

        // ROS_INFO("Thrust: %f, Roll: %f, Pitch: %f, Yaw Rate: %f", ctrl(0), ctrl(1), ctrl(2), ctrl(3));
        control_pub_.publish(msg);
        
        rate.sleep();
        ros::spinOnce();
    }
}

