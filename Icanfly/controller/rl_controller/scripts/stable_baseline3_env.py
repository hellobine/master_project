#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
import torch
import rospy
import threading
import random

from gymnasium import spaces
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker


# --------------------------------------------------
# Helper
# --------------------------------------------------

def compute_desired_state(traj_time: float, T: float, origin_offset: float):
    """8‑shape trajectory (3.5 s period) and its first derivative."""
    x = np.cos(2 * np.pi * traj_time / T)
    y = np.sin(4 * np.pi * traj_time / T) / 2.0
    z = 1.0
    desired_pos = np.array([origin_offset + x, y, z], dtype=np.float32)

    dx = -(2 * np.pi / T) * np.sin(2 * np.pi * traj_time / T)
    dy =  (4 * np.pi / T) * np.cos(4 * np.pi * traj_time / T) / 2.0
    dz = 0.0
    desired_vel = np.array([dx, dy, dz], dtype=np.float32)
    return desired_pos, np.array([1, 0, 0, 0], np.float32), desired_vel


# --------------------------------------------------
# Environment
# --------------------------------------------------

class QuadrotorEnv(gym.Env):
    """Continuous‑control quadrotor environment compatible with Gymnasium."""

    # FUTURE_STEPS = 5                    # prediction horizon for obs
    def __init__(self, namespace: str = "drone"):
        super().__init__()
        self.namespace = namespace
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        # ---- Physical constants -------------------------------------
        self.mass = 0.73
        self.gravity = 9.8066
        self.min_thrust = 0.5 * self.mass * self.gravity
        self.max_thrust = 5.0 * self.mass * self.gravity
        self.max_angular_rate = 5.0               # rad/s


        self.FUTURE_STEPS = 5

        # ---- Origin offset (multi‑drone support) --------------------
        self.origin_offset = -5.0
        for i in range(5):
            if str(i) in namespace:
                self.origin_offset += 2.0 * i

        # ---- Episode bookkeeping -----------------------------------
        self.max_episode_steps = 1000
        self.step_count = 0
        self.episode_reward = 0.0

        # ---- Reward weights ----------------------------------------
        # up
        self.reward_up_weight = 1.0
        # spin
        self.reward_spin_weight = 1.0
        # acc
        self.reward_acc_weight_init = 0.0
        self.reward_acc_weight_lr= 0.0005 # slow= 0.0001, fast: 0.0005
        self.reward_acc_max= 0.0
        # jerk
        self.reward_jerk_weight_init= 0.0
        self.reward_jerk_weight_lr= 0.0005 # slow: 0.0001, fast: 0.0005
        self.reward_jerk_max= 0.0
        # snap
        self.reward_snap_weight_init= 0.0
        self.reward_snap_weight_lr= 0.0005 # slow= 0.0001, fast= 0.0005
        self.reward_snap_max= 0.0
        # action smoothness
        self.reward_action_smoothness_weight_init= 2.0
        self.reward_action_smoothness_weight_lr= 0.0005 # slow= 0.0001, fast= 0.0005
        self.reward_smoothness_max= 2.0
        # action norm
        self.reward_action_norm_weight_init= 0.0
        self.reward_action_norm_weight_lr= 0.0001 # slow= 0.0001, fast= 0.0005
        self.reward_norm_max= 0.0
        # distance
        self.reward_distance_scale= 5.0

        # ---- Dynamics history --------------------------------------
        self.action_dim = 4
        self.prev_action = np.zeros(self.action_dim, np.float32)
        self.last_linear_v = 0.0
        self.last_angular_v = 0.0
        self.last_linear_a = 0.0
        self.last_angular_a = 0.0
        self.last_linear_jerk = 0.0

        # ---- Safety bounds (also used for scaling) -----------------
        self.max_position_error  = 4      # m
        self.max_velocity_error  = 6.0      # m/s

        # ---- ROS ----------------------------------------------------
        if not rospy.core.is_initialized():
            rospy.init_node(f"{namespace}_quad_env", anonymous=True)
        self.control_hz = 100.0
        self.control_dt = 1.0 / self.control_hz
        self.rate = rospy.Rate(self.control_hz)

        self._odom_lock = threading.Lock()
        self.odom_sub = rospy.Subscriber(
            f"/{namespace}/ground_truth/odometry", Odometry, self._odom_cb)

        self.cmd_pub  = rospy.Publisher(f"/{namespace}/control_command", ControlCommand, queue_size=1)
        self.arm_pub  = rospy.Publisher(f"/{namespace}/bridge/arm", Bool, queue_size=1)
        self.point_pub = rospy.Publisher(f"/{namespace}/desired_point_marker", Marker, queue_size=1)

        # ---- Action/Observation spaces ------------------------------
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        obs_low  = np.concatenate([
            -self.max_position_error * np.ones(self.FUTURE_STEPS * 3),
            -np.ones(9),
            -self.max_velocity_error * np.ones(3),
            -self.max_angular_rate  * np.ones(3),
            -np.ones(4)
        ]).astype(np.float32)

        obs_high = np.abs(obs_low)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=obs_low.shape, dtype=np.float32)

        # Pre‑store scaling factor for normalization
        self._obs_scale = obs_high

        # ---- Trajectory --------------------------------------------
        self.T            = 3.5           # sec per loop
        self._traj_time   = 0.0
        self.des_pos        = np.zeros(3, np.float32)
        self._last_step_time = rospy.Time.now()

        # Arm after a short delay
        rospy.sleep(1.0)
        if not rospy.is_shutdown():
            self.arm_pub.publish(Bool(data=True))

    # -------------------------------------------------- ROS callbacks
    def _odom_cb(self, msg: Odometry):
        with self._odom_lock:
            pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            ori = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                   msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
            lin = [msg.twist.twist.linear.x,  msg.twist.twist.linear.y,  msg.twist.twist.linear.z]
            ang = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
            self.current_state = np.array(pos + ori + lin + ang, dtype=np.float32)

    # -------------------------------------------------- Public API
    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)  # ensure bound

        # ----- send command ----------------------------------------
        now = rospy.Time.now()
        dt = (now - self._last_step_time).to_sec(); self._last_step_time = now
        self._traj_time += dt
        self._publish_desired_marker()

        cmd = ControlCommand()
        cmd.armed = True;  cmd.control_mode = ControlCommand.BODY_RATES
        cmd.collective_thrust = ((action[0] + 1.0) / 2.0) * (self.max_thrust - self.min_thrust) + self.min_thrust
        cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = action[1:] * self.max_angular_rate
        self.cmd_pub.publish(cmd)

        # ----- observe --------------------------------------------
        with self._odom_lock:
            state = self.current_state.copy()
 
        obs, reward = self._build_obs_and_reward(state, action)
 
        # ----- update episode bookkeeping ------------------------
        self.step_count += 1
        self.episode_reward += reward
        done = self._check_done(state) or self.step_count >= self.max_episode_steps

        info = {"reward": reward}
        if done:
            info["episode"] = {"r": self.episode_reward, "l": self.step_count}
            self.step_count = 0; self.episode_reward = 0.0
        return obs, reward, done, False, info

    def reset(self, **kwargs):
        self.step_count = 0; self.episode_reward = 0.0; self.prev_action.fill(0.0)
        self._traj_time = 0.0; self._last_step_time = rospy.Time.now()

        # place drone near origin offset
        reset_pos = np.array([
            self.origin_offset ,
            0, 1
        ], np.float32)
        self._reset_pose(reset_pos)

        with self._odom_lock:
            state = self.current_state.copy()
        obs, _ = self._build_obs_and_reward(state, self.prev_action)
        return obs, {}

    def render(self, mode="human"):
        pass  # RViz does the visualization

    def close(self):
        rospy.signal_shutdown("Environment closed")

    # -------------------------------------------------- Internals
    def _publish_desired_marker(self):
        marker = Marker()
        marker.header.frame_id = "world"; marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0, 1, 0, 1
        
        dp, _, _ = compute_desired_state(self._traj_time, self.T, self.origin_offset)
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = dp.astype(float)
        self.point_pub.publish(marker)

    def _reset_pose(self, p: np.ndarray):
        pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        timeout = rospy.Duration(10.0); start = rospy.Time.now()
        while not rospy.is_shutdown():
            m = ModelState(); m.model_name = self.namespace
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p
            m.pose.orientation.w = 1.0
            # m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = np.random.uniform(-0.2, 0.2, 3)
            # m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = np.random.uniform(-0.1, 0.1, 3)
            pub.publish(m)
            with self._odom_lock:
                if np.linalg.norm(self.current_state[0:3] - p) < 0.1 or rospy.Time.now() - start > timeout:
                    break

    # ----------------------- Observation & reward ------------------
    def _build_obs_and_reward(self, state: np.ndarray, action: np.ndarray):
        # device = self.device
        # ---------------------------------------------------------------------
        # 1.  Tensor helpers (pos / vel / rot)
        # ---------------------------------------------------------------------
        pos      = torch.tensor(state[0:3],  device=self.device)
        quat     = torch.tensor(state[3:7],  device=self.device)
        lin_vel  = torch.tensor(state[7:10], device=self.device)
        ang_vel  = torch.tensor(state[10:13], device=self.device)

         # rotation matrix, up‑vector
        rot_mat  = torch.tensor(self.quaternion_to_rot_matrix(quat.cpu().numpy()), device=self.device).view(3, 3)
        up_z     = rot_mat[2, 2]

        # ---------------------------------------------------------------------
        # 2.  Future waypoint errors  (K = FUTURE_STEPS).
        #     Same as Track:   rpos = (target_pos[0:K] - current_pos).flatten()
        # ---------------------------------------------------------------------
        t_idx    = torch.arange(self.FUTURE_STEPS, device=self.device)            # 0..K‑1
        t_query  = self._traj_time + (t_idx + 1) * self.control_dt           # (+1) 与 Track 对齐: 下一个物理步开始

        # 查询参考轨迹（向量化实现）
        desired_pos = []
        for t_i in t_query.tolist():
            p_i, _, _ = compute_desired_state(t_i, self.T, self.origin_offset)
            desired_pos.append(torch.tensor(p_i, device=self.device))
        desired_pos = torch.stack(desired_pos)                                # (K,3)

        future_err  = (desired_pos - pos).flatten()                           # (3K,)

        # ---------------------------------------------------------------------
        # 3.  Rewards  ---------------------------------------------------------
        # ---------------------------------------------------------------------
        # 3.1  Main distance term (only the 0‑th waypoint)
        distance        = torch.norm(future_err[0:3])
        reward_pos      = self.reward_distance_scale * torch.exp(-distance)

        # 3.2  Uprightness & spin  (gated by reward_pos later)
        tiltage         = torch.abs(1.0 - up_z)
        reward_up       = self.reward_up_weight   * 0.5 / (1.0 + tiltage ** 2)

        omega_z         = ang_vel[-1]
        spin_sq         = omega_z ** 2
        reward_spin     = self.reward_spin_weight * 0.5 / (1.0 + spin_sq ** 2)

        # 3.3  Action regularisation ------------------------------------------
        a_tensor        = torch.tensor(action, device=self.device)

        # --- (i) Action norm --------------------------------------------------
        w_norm = min(self.reward_action_norm_weight_init +
                        self.reward_action_norm_weight_lr * self.step_count,
                        self.reward_norm_max)
        reward_norm = w_norm * torch.exp(-torch.norm(a_tensor))

        # --- (ii) Action smoothness ------------------------------------------
        if self.step_count == 0:
            reward_smooth = torch.tensor(0.0, device=self.device)
            delta_a_norm  = torch.tensor(0.0, device=self.device)
        else:
            delta_a       = a_tensor - torch.tensor(self.prev_action, device=self.device)
            delta_a_norm  = torch.norm(delta_a)
            w_smooth = min(self.reward_action_smoothness_weight_init +
                            self.reward_action_smoothness_weight_lr * self.step_count,
                            self.reward_smoothness_max)
            reward_smooth = w_smooth * torch.exp(-delta_a_norm)

        # 3.4  Dynamics regularisation (acc / jerk / snap) --------------------
        linear_v     = torch.norm(lin_vel)
        if self.step_count < 2:                 # 先收两帧再算
            linear_a = torch.tensor(0., device=self.device)
            linear_jerk = torch.tensor(0., device=self.device)
            linear_snap = torch.tensor(0., device=self.device)
        else:
            linear_v = torch.norm(lin_vel)
            linear_a = torch.abs(linear_v - self.last_linear_v) / self.control_dt
            linear_jerk = torch.abs(linear_a - self.last_linear_a) / self.control_dt
            linear_snap = torch.abs(linear_jerk - self.last_linear_jerk) / self.control_dt

 
        w_acc  = min(self.reward_acc_weight_init  + self.reward_acc_weight_lr  * self.step_count, self.reward_acc_max)
        w_jerk = min(self.reward_jerk_weight_init + self.reward_jerk_weight_lr * self.step_count, self.reward_jerk_max)
        w_snap = min(self.reward_snap_weight_init + self.reward_snap_weight_lr * self.step_count, self.reward_snap_max)

        reward_acc  = w_acc  * torch.exp(-linear_a)
        reward_jerk = w_jerk * torch.exp(-linear_jerk)
        reward_snap = w_snap * torch.exp(-linear_snap)
 
        # 3.5  Final aggregation ----------------------------------------------
        reward_total = (
            reward_pos +
            reward_pos * (reward_up + reward_spin) +
            reward_norm + reward_smooth 
            # + reward_acc  + reward_jerk   + reward_snap
        )

        # 3.5 Aggregate
        term_pos        = reward_pos
        term_up         = reward_pos * reward_up
        term_spin       = reward_pos * reward_spin
        term_norm       = reward_norm
        term_smooth     = reward_smooth
        term_acc        = reward_acc
        term_jerk       = reward_jerk
        term_snap       = reward_snap

        reward = reward_total.item()

        # print("========== REWARD BREAKDOWN ==========")
        # print(f"distance          : {distance.item():.4f}")
        # print(f"term_pos          : {term_pos.item():.4f}")
        # print(f"term_up           : {term_up.item():.4f}")
        # print(f"term_spin         : {term_spin.item():.4f}")
        # print(f"term_norm (w={w_norm:.3f}) : {term_norm.item():.4f}")
        # print(f"term_smooth       : {term_smooth.item():.4f}")
        # print(f"term_acc          : {term_acc.item():.4f}")
        # print(f"term_jerk         : {term_jerk.item():.4f}")
        # print(f"term_snap         : {term_snap.item():.4f}")
        # print(f"TOTAL reward      : {reward:.4f}\n")

        # ---- update history -------------------------------------
        self.prev_action      = action.copy()
        self.last_linear_jerk = linear_jerk.item()
        self.last_linear_a    = linear_a.item()
        self.last_linear_v    = linear_v.item()
        self.last_angular_v   = torch.norm(ang_vel).item()

        obs_parts = [future_err, rot_mat.flatten(), lin_vel, ang_vel,
+              torch.tensor(action, device=self.device)]
        obs_raw = torch.cat(obs_parts)  # (obs_dim,)

        # normalise to [‑1,1] with predefined scale vector (same as Track)
        obs = torch.clamp(obs_raw / torch.tensor(self._obs_scale, device=self.device), -1.0, 1.0).cpu().numpy().astype(np.float32)

        return obs, reward


    def _check_done(self, state):
        pos = np.array(state[0:3])
        t_i = self._traj_time
        dp, _, _ = compute_desired_state(t_i, self.T, self.origin_offset)
        pos_error = np.linalg.norm(dp - pos)
        if pos[2] < 0.3 or pos_error > self.max_position_error:
            return True
        qw, qx, qy, qz = state[3:7]
        roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
        if abs(roll) > 2.2 or np.linalg.norm(state[7:10]) > self.max_velocity_error:
            return True
        return False

    def render(self, mode='human'):
        pass
    
    def quaternion_to_rot_matrix(self, quat):
        """
        Convert quaternion (w, x, y, z) to flattened 3x3 rotation matrix.
        """
        qw, qx, qy, qz = quat
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,       1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,       2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
        ], dtype=np.float32)
        return R.flatten()
    
    def close(self):
        rospy.signal_shutdown("Environment closed")


