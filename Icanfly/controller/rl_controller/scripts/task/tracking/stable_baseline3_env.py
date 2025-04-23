# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import gymnasium as gym
# import numpy as np
# import torch
# import rospy
# import threading
# import random

# from gymnasium import spaces
# from geometry_msgs.msg import Vector3
# from nav_msgs.msg import Odometry
# from quadrotor_msgs.msg import ControlCommand
# from gazebo_msgs.msg import ModelState
# from std_msgs.msg import Bool
# from visualization_msgs.msg import Marker

# from typing import Optional, Tuple, Union

# # --------------------------------------------------
# # Helper
# # --------------------------------------------------

# # def compute_desired_state(traj_time: float, T: float, origin_offset: float):
# #     """8‑shape trajectory (3.5 s period) and its first derivative."""
# #     x = np.cos(2 * np.pi * traj_time / T)
# #     y = np.sin(4 * np.pi * traj_time / T) / 2.0
# #     z = 1.0
# #     desired_pos = np.array([origin_offset + x, y, z], dtype=np.float32)

# #     dx = -(2 * np.pi / T) * np.sin(2 * np.pi * traj_time / T)
# #     dy =  (4 * np.pi / T) * np.cos(4 * np.pi * traj_time / T) / 2.0
# #     dz = 0.0
# #     desired_vel = np.array([dx, dy, dz], dtype=np.float32)
# #     return desired_pos, np.array([1, 0, 0, 0], np.float32), desired_vel

# def compute_desired_state(
#     traj_time: Union[float, torch.Tensor],
#     T: float,
#     origin_offset: float,
#     *,
#     device: Optional[Union[torch.device, str]] = None,
#     dtype: torch.dtype = torch.float32,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     8-字轨迹 + 一阶导数   (完全 torch 化)
#     返回:
#         desired_pos : (3,) Tensor  [x, y, z]
#         quat_wxyz   : (4,) Tensor  恒 [1,0,0,0]  (水平朝向)
#         desired_vel : (3,) Tensor  [dx, dy, dz]
#     """
#     if device is None:
#         device = "cpu"

#     # 把标量时间包装成张量，方便广播/梯度
#     t = torch.as_tensor(traj_time, dtype=dtype, device=device)

#     two_pi_over_T = 2.0 * torch.pi / T

#     # 轨迹本身
#     x = torch.cos(two_pi_over_T * t)
#     y = 0.5 * torch.sin(2.0 * two_pi_over_T * t)
#     z = torch.ones_like(t)                         # 常量 1

#     desired_pos = torch.stack([origin_offset + x, y, z], dim=-1)  # (..., 3)

#     # 一阶导数
#     dx = -two_pi_over_T * torch.sin(two_pi_over_T * t)
#     dy =  2.0 * two_pi_over_T * torch.cos(2.0 * two_pi_over_T * t) * 0.5
#     dz = torch.zeros_like(t)

#     desired_vel = torch.stack([dx, dy, dz], dim=-1)               # (..., 3)

#     quat_wxyz = torch.tensor([1, 0, 0, 0], dtype=dtype,
#                              device=device).expand(desired_pos.shape[:-1] + (4,))

#     return desired_pos, quat_wxyz, desired_vel

# def quaternion_to_rot_matrix(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     """(…,4) → (…,3,3)"""
#     quat = torch.as_tensor(quat)
#     w, x, y, z = quat.unbind(-1)
#     norm = torch.sqrt(w * w + x * x + y * y + z * z + eps)
#     w, x, y, z = w / norm, x / norm, y / norm, z / norm
#     ww, xx, yy, zz = w * w, x * x, y * y, z * z
#     xy, xz, yz = x * y, x * z, y * z
#     wx, wy, wz = w * x, w * y, w * z
#     R = torch.stack(
#         [
#             torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
#             torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
#             torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
#         ],
#         dim=-2,
#     )
#     return R


# def safe_normalize(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     norm = torch.linalg.norm(q)
#     norm = norm if norm > eps else torch.tensor(eps, device=q.device)
#     return q / norm

# # --------------------------------------------------
# # Environment
# # --------------------------------------------------

# class QuadrotorEnv(gym.Env):
#     """Continuous‑control quadrotor environment compatible with Gymnasium."""

#     # FUTURE_STEPS = 5                    # prediction horizon for obs
#     def __init__(self, namespace: str = "drone"):
#         super().__init__()
#         self.namespace = namespace
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         # ---- Physical constants -------------------------------------
#         self.mass = 0.73
#         self.gravity = 9.8066
#         self.min_thrust = 0.5 * self.mass * self.gravity
#         self.max_thrust = 5.0 * self.mass * self.gravity
#         self.max_angular_rate = 5.0               # rad/s


#         self.FUTURE_STEPS = 10

#         # ---- Origin offset (multi‑drone support) --------------------
#         self.origin_offset = -5.0
#         for i in range(10):
#             if str(i) in namespace:
#                 self.origin_offset += 3.0 * i

#         # ---- Episode bookkeeping -----------------------------------
#         self.max_episode_steps = 256
#         self.step_count = 0
#         self.episode_reward = 0.0

#         # ---- Reward weights ----------------------------------------
#         self.s_angle_diff = -1.0
#         self.s_target = 0.5
#         # up
#         self.reward_up_weight = 1.0
#         # spin
#         self.reward_spin_weight = 1.0
#         # acc
#         self.reward_acc_weight_init = 0.0
#         self.reward_acc_weight_lr= 0.0005 # slow= 0.0001, fast: 0.0005
#         self.reward_acc_max= 0.0
#         # jerk
#         self.reward_jerk_weight_init= 0.0
#         self.reward_jerk_weight_lr= 0.0005 # slow: 0.0001, fast: 0.0005
#         self.reward_jerk_max= 0.0
#         # snap
#         self.reward_snap_weight_init= 0.0
#         self.reward_snap_weight_lr= 0.0005 # slow= 0.0001, fast= 0.0005
#         self.reward_snap_max= 0.0
#         # action smoothness
#         self.reward_action_smoothness_weight_init= 0.4
#         self.reward_action_smoothness_weight_lr= 0.0005 # slow= 0.0001, fast= 0.0005
#         self.reward_smoothness_max= 1.0
#         # action norm
#         self.reward_action_norm_weight_init= 0.0
#         self.reward_action_norm_weight_lr= 0.0001 # slow= 0.0001, fast= 0.0005
#         self.reward_norm_max= 0.0
#         # distance
#         self.reward_distance_scale= 5.0
#         self.s_crash = -10

#         # ---- Dynamics history --------------------------------------
#         self.action_dim = 4
#         self.prev_action = torch.zeros(self.action_dim, device=self.device)
        
#         self.last_linear_v = 0.0
#         self.last_angular_v = 0.0
#         self.last_linear_a = 0.0
#         self.last_angular_a = 0.0
#         self.last_linear_jerk = 0.0
#         self.prev_orientation_q = torch.tensor(
#             [1.0, 0.0, 0.0, 0.0],
#             dtype=torch.float32,
#             device=self.device
#         )
#         self.prev_position = torch.tensor(
#             [0.0, 0.0, 0.0],
#             dtype=torch.float32,
#             device=self.device
#         )
#         # self.prev_orientation_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
#         # self.prev_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
#         # ---- Safety bounds (also used for scaling) -----------------
#         self.max_position_error  = 2.5      # m
#         self.max_velocity_error  = 6.0      # m/s

#         # ---- ROS ----------------------------------------------------
#         if not rospy.core.is_initialized():
#             rospy.set_param('use_sim_time', True)
#             rospy.init_node(f"{namespace}_quad_env", anonymous=True)
#         self.control_hz = 100.0
#         self.control_dt = 1.0 / self.control_hz
#         self.rate = rospy.Rate(self.control_hz)

#         self._odom_lock = threading.Lock()
#         self.odom_sub = rospy.Subscriber(
#             f"/{namespace}/ground_truth/odometry", Odometry, self._odom_cb)

#         self.cmd_pub  = rospy.Publisher(f"/{namespace}/control_command", ControlCommand, queue_size=1)
#         self.arm_pub  = rospy.Publisher(f"/{namespace}/bridge/arm", Bool, queue_size=1)
#         self.point_pub = rospy.Publisher(f"/{namespace}/desired_point_marker", Marker, queue_size=1)

#         # ---- Action/Observation spaces ------------------------------
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

#         obs_low  = np.concatenate([
#             -self.max_position_error * np.ones(self.FUTURE_STEPS * 3),
#             -np.ones(9),
#             -self.max_velocity_error * np.ones(3),
#             -self.max_angular_rate  * np.ones(3),
#             -np.ones(4)
#         ]).astype(np.float32)

#         obs_high = np.abs(obs_low)

#         self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=obs_low.shape, dtype=np.float32)

#         # Pre‑store scaling factor for normalization
#         self._obs_scale = obs_high

#         # ---- Trajectory --------------------------------------------
#         self.T            = 10           # sec per loop
#         self._traj_time   = 0.0
#         self.des_pos        = np.zeros(3, np.float32)
#         self._last_step_time = rospy.Time.now()

#         # Arm after a short delay
#         rospy.sleep(1.0)
#         if not rospy.is_shutdown():
#             self.arm_pub.publish(Bool(data=True))

#     # -------------------------------------------------- ROS callbacks
#     def _odom_cb(self, msg: Odometry):
#         with self._odom_lock:
#             # pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
#             # ori = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
#             #        msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
#             # lin = [msg.twist.twist.linear.x,  msg.twist.twist.linear.y,  msg.twist.twist.linear.z]
#             # ang = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
#             # self.current_state = np.array(pos + ori + lin + ang, dtype=np.float32)
#             pos = torch.tensor([
#                 msg.pose.pose.position.x,
#                 msg.pose.pose.position.y,
#                 msg.pose.pose.position.z,
#             ], device=self.device)
#             ori = torch.tensor([
#                 msg.pose.pose.orientation.w,
#                 msg.pose.pose.orientation.x,
#                 msg.pose.pose.orientation.y,
#                 msg.pose.pose.orientation.z,
#             ], device=self.device)
#             lin = torch.tensor([
#                 msg.twist.twist.linear.x,
#                 msg.twist.twist.linear.y,
#                 msg.twist.twist.linear.z,
#             ], device=self.device)
#             ang = torch.tensor([
#                 msg.twist.twist.angular.x,
#                 msg.twist.twist.angular.y,
#                 msg.twist.twist.angular.z,
#             ], device=self.device)
#             self.current_state = torch.cat([pos, ori, lin, ang])  # (13,)

#     # -------------------------------------------------- Public API
#     def step(self, action):
#         action = np.clip(action, -1.0, 1.0).astype(np.float32)  # ensure bound

#         # ----- send command ----------------------------------------
#         now = rospy.Time.now()
#         dt = (now - self._last_step_time).to_sec()
#         # print(f"dt: {dt}")
#         self._last_step_time = now
#         self._traj_time += dt
#         self._publish_desired_marker()

#         cmd = ControlCommand()
#         cmd.armed = True;  cmd.control_mode = ControlCommand.BODY_RATES
#         cmd.collective_thrust = ((action[0] + 1.0) / 2.0) * (self.max_thrust - self.min_thrust) + self.min_thrust
#         cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = action[1:] * self.max_angular_rate
#         self.cmd_pub.publish(cmd)
#         # print(f"Rospy.time: {rospy.Time.now().to_sec()}")
#         rospy.wait_for_message(f"/{self.namespace}/ground_truth/odometry", Odometry)
   
#         # print(f"Rospy.time: {rospy.Time.now().to_sec()}")
#         # ----- observe --------------------------------------------
#         with self._odom_lock:
#             state = self.current_state.clone()
 
#         obs, reward = self._build_obs_and_reward(state, torch.from_numpy(action).to(self.device))
 
#         # ----- update episode bookkeeping ------------------------
#         self.step_count += 1
#         self.episode_reward += reward

#         # print(f"step: {self.step_count}, self.episode_reward: {self.episode_reward}")
#         done = self._check_done(state) or self.step_count >= self.max_episode_steps

#         info = {"reward": reward}
#         # print(f"step: {self.step_count}, reward: {reward:.4f}, done: {done}")
#         if done:
#             info["episode"] = {"r": self.episode_reward, "l": self.step_count}
#             self.step_count = 0; self.episode_reward = 0.0
#         return obs, reward, done, False, info

#     def reset(self, **kwargs):
#         self.step_count = 0; 
#         self.episode_reward = 0.0; 
#         # self.prev_action.fill(0.0)
#         self.prev_action.zero_()
#         self._traj_time = 0.0; 
#         self._last_step_time = rospy.Time.now()
#         self.prev_orientation_q = torch.tensor(
#             [1.0, 0.0, 0.0, 0.0],
#             dtype=torch.float32,
#             device=self.device
#         )
#         self.prev_position = torch.tensor(
#             [self.origin_offset + 1, 0, 1],
#             dtype=torch.float32,
#             device=self.device
#         )

#         # self.prev_orientation_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
#         # self.prev_position = torch.tensor([ self.origin_offset + 1, 0, 1], dtype=torch.float32)
#         self.T = random.randint(3, 15)

#         # place drone near origin offset
#         # reset_pos = torch.tensor([
#         #     self.origin_offset + np.random.uniform(-1, 1),
#         #     np.random.uniform(-1, 1), np.random.uniform(0.6, 1.8)
#         # ], torch.float32)
#         reset_pos = torch.stack([
#             torch.tensor(self.origin_offset, device=self.device) + (torch.rand(1, device=self.device)*2 - 1),
#             torch.rand(1, device=self.device)*2 - 1,
#             torch.rand(1, device=self.device)*1.2 + 0.6
#         ]).view(3)

#         self._reset_pose(reset_pos)

#         with self._odom_lock:
#             state = self.current_state.clone()
#         obs, _ = self._build_obs_and_reward(state, self.prev_action)
#         return obs, {}

#     def render(self, mode="human"):
#         pass  # RViz does the visualization

#     def close(self):
#         rospy.signal_shutdown("Environment closed")

#     # -------------------------------------------------- Internals
#     def _publish_desired_marker(self):
#         marker = Marker()
#         marker.header.frame_id = "world"; marker.header.stamp = rospy.Time.now()
#         marker.type = Marker.SPHERE; marker.action = Marker.ADD
#         marker.scale.x = marker.scale.y = marker.scale.z = 0.1
#         marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0, 1, 0, 1
        
#         dp, _, _ = compute_desired_state(self._traj_time, self.T, self.origin_offset, device=self.device)
#         p_np = dp.detach().cpu().numpy()        # now a numpy array
#         marker.pose.position.x = float(p_np[0])
#         marker.pose.position.y = float(p_np[1])
#         marker.pose.position.z = float(p_np[2])
#         # marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = dp.astype(float)
#         self.point_pub.publish(marker)

#     # def _reset_pose(self, p: Union[torch.Tensor]):
#     #     pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
#     #     timeout = rospy.Duration(10.0); start = rospy.Time.now()
#     #     while not rospy.is_shutdown():
#     #         m = ModelState(); m.model_name = self.namespace
#     #         m.pose.position.x, m.pose.position.y, m.pose.position.z = p
#     #         m.pose.orientation.w = 1.0
#     #         m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = np.random.uniform(-0.2, 0.2, 3)
#     #         m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = np.random.uniform(-0.1, 0.1, 3)
#     #         pub.publish(m)
#     #         with self._odom_lock:
#     #             if np.linalg.norm(self.current_state[0:3] - p) < 0.1 or rospy.Time.now() - start > timeout:
#     #                 break

#     def _reset_pose(self, p: torch.Tensor):
#         """
#         Reset the drone pose to p (a torch.Tensor on cpu or cuda of shape (3,)).
#         Uses torch for everything, only .item() when assigning to ROS messages.
#         """
#         # 确保 p 在正确设备并是 Tensor
#         if not isinstance(p, torch.Tensor):
#             p = torch.as_tensor(p, device=self.device)
#         else:
#             p = p.to(self.device)

#         pub     = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
#         timeout = rospy.Duration(10.0)
#         start   = rospy.Time.now()

#         while not rospy.is_shutdown():
#             # 构造消息
#             m = ModelState()
#             m.model_name = self.namespace
#             # 直接用 Tensor.item()
#             m.pose.position.x = p[0].item()
#             m.pose.position.y = p[1].item()
#             m.pose.position.z = p[2].item()
#             m.pose.orientation.w = 1.0

#             # 随机线性/角速度：torch.rand → Python 列表
#             lin_rand = (torch.rand(3, device=self.device) * 0.4 - 0.2).tolist()
#             ang_rand = (torch.rand(3, device=self.device) * 0.2 - 0.1).tolist()
#             m.twist.linear.x, m.twist.linear.y, m.twist.linear.z   = lin_rand
#             m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = ang_rand

#             pub.publish(m)

#             # 用 torch 计算当前状态到目标的距离
#             with self._odom_lock:
#                 dist = torch.linalg.norm(self.current_state[:3] - p)

#             # 条件满足则退出
#             if dist.item() < 0.1 or (rospy.Time.now() - start) > timeout:
#                 break

#     # def safe_normalize(self, q, epsilon=1e-6):
#     #     """
#     #     安全归一化四元数，防止除以零
#     #     """
#     #     # q = np.array(q)  # 转换为 NumPy 数组
#     #     # norm = np.linalg.norm(q)
#     #     q = torch.as_tensor(q, device=self.device)
#     #     norm = torch.linalg.norm(q)
#     #     if norm < epsilon:
#     #         norm = epsilon
#     #     return q / norm
    
    

#     # ----------------------- Observation & reward ------------------
#     def _build_obs_and_reward(self, state: torch.Tensor, action: torch.Tensor):
#         # device = self.device
#         # ---------------------------------------------------------------------
#         # 1.  Tensor helpers (pos / vel / rot)
#         # ---------------------------------------------------------------------
#         # pos      = torch.tensor(state[0:3],  device=self.device)
#         # quat     = torch.tensor(state[3:7],  device=self.device)
#         # lin_vel  = torch.tensor(state[7:10], device=self.device)
#         # ang_vel  = torch.tensor(state[10:13], device=self.device)
#         pos     = state[0:3].to(self.device)
#         quat    = state[3:7].to(self.device)
#         lin_vel = state[7:10].to(self.device)
#         ang_vel = state[10:13].to(self.device)
#         # a_tensor = torch.as_tensor(action,      device=self.device)



#          # rotation matrix, up‑vector
#         rot_mat_np = quaternion_to_rot_matrix(quat)   # 继续复用旧函数
#         rot_mat    = torch.as_tensor(rot_mat_np, device=self.device).view(3, 3)
#         up_z       = rot_mat[2, 2]

#         # ---------------------------------------------------------------------
#         # 2.  Future waypoint errors  (K = FUTURE_STEPS).
#         #     Same as Track:   rpos = (target_pos[0:K] - current_pos).flatten()
#         # ---------------------------------------------------------------------
#         t_idx    = torch.arange(self.FUTURE_STEPS, device=self.device)            # 0..K‑1
#         t_query  = self._traj_time + (t_idx + 1) * self.control_dt           # (+1) 与 Track 对齐: 下一个物理步开始

#         # 查询参考轨迹（向量化实现）
#         desired_pos = []
#         for t_i in t_query.tolist():
#             p_i, _, _ = compute_desired_state(t_i, self.T, self.origin_offset)
#             desired_pos.append(torch.tensor(p_i, device=self.device))
#         desired_pos = torch.stack(desired_pos)                                # (K,3)
#         future_err  = (desired_pos - pos).flatten()                           # (3K,)
#         # print(f"future_err: {future_err}")
#         # ---------------------------------------------------------------------
#         # 3.  Rewards  ---------------------------------------------------------
#         # ---------------------------------------------------------------------
#         # 3.1  Main distance term (only the 0‑th waypoint)
#         # distance        = torch.norm(future_err[0:3])
#         # curr_rel_dis = #np.clip(np.linalg.norm(future_err[0:3])/self.max_position_error,0,1)
#         curr_rel_dis = torch.clamp(torch.linalg.norm(future_err[0:3])/self.max_position_error, min=0.0, max=1.0)
#         reward_pos = self.reward_distance_scale * torch.exp(-curr_rel_dis)

#         # print(f"reward_pos: {reward_pos.item():.4f}")


#         prev_dist = torch.linalg.norm(self.prev_position - desired_pos[0:3])
#         curr_dist = torch.linalg.norm(pos - desired_pos[0:3])
#         r_target = torch.tanh(30*(prev_dist - curr_dist))
#         # print(f"r_target: {r_target:.4f}")


#         # ------------------------------------------------------------------ #
#         # 3-2. 姿态差异                                                      #
#         # ------------------------------------------------------------------ #
#         epsilon = 1e-6
#         q_prev  = safe_normalize(
#             torch.as_tensor(self.prev_orientation_q, device=self.device), epsilon
#         )
#         q_curr  = safe_normalize(quat, epsilon)

#         dot_product = torch.clamp(
#             torch.dot(q_prev, q_curr), -1.0, 1.0
#         )
#         angle_diff  = 2.0 * torch.arccos(torch.abs(dot_product))



#         # 3.2  Uprightness & spin  (gated by reward_pos later)
#         tiltage         = torch.abs(1.0 - up_z)
#         reward_up       = self.reward_up_weight   * 0.5 / (1.0 + tiltage ** 2)

#         omega_z         = ang_vel[-1]
#         spin_sq         = omega_z ** 2
#         reward_spin     = self.reward_spin_weight * 0.5 / (1.0 + spin_sq ** 2)

#         # 3.3  Action regularisation ------------------------------------------
#         # a_tensor        = torch.tensor(action, device=self.device)

#         # --- (i) Action norm --------------------------------------------------
#         w_norm = min(self.reward_action_norm_weight_init +
#                         self.reward_action_norm_weight_lr * self.step_count,
#                         self.reward_norm_max)
#         reward_norm = w_norm * torch.exp(-torch.norm(action))

#         # --- (ii) Action smoothness ------------------------------------------
#         if self.step_count == 0:
#             reward_smooth = torch.tensor(0.0, device=self.device)
#             delta_a_norm  = torch.tensor(0.0, device=self.device)
#         else:
#             delta_a       = action - self.prev_action #torch.tensor(self.prev_action, device=self.device)
#             delta_a_norm  = torch.norm(delta_a)/5.0
#             # w_smooth = min(self.reward_action_smoothness_weight_init +
#             #                 self.reward_action_smoothness_weight_lr * self.step_count,
#             #                 self.reward_smoothness_max)
 
#             reward_smooth = torch.clamp(-0.5 * delta_a_norm, min=-1.0, max=1.0)
#             # print(f"reward_smooth: {reward_smooth.item():.4f}")

#         # 3.4  Dynamics regularisation (acc / jerk / snap) --------------------
#         linear_v     = torch.norm(lin_vel)
#         if self.step_count < 2:                 # 先收两帧再算
#             linear_a = torch.tensor(0., device=self.device)
#             linear_jerk = torch.tensor(0., device=self.device)
#             linear_snap = torch.tensor(0., device=self.device)
#         else:
#             linear_v = torch.norm(lin_vel)
#             linear_a = torch.abs(linear_v - self.last_linear_v) / self.control_dt
#             linear_jerk = torch.abs(linear_a - self.last_linear_a) / self.control_dt
#             linear_snap = torch.abs(linear_jerk - self.last_linear_jerk) / self.control_dt

 
#         w_acc  = min(self.reward_acc_weight_init  + self.reward_acc_weight_lr  * self.step_count, self.reward_acc_max)
#         w_jerk = min(self.reward_jerk_weight_init + self.reward_jerk_weight_lr * self.step_count, self.reward_jerk_max)
#         w_snap = min(self.reward_snap_weight_init + self.reward_snap_weight_lr * self.step_count, self.reward_snap_max)

#         reward_acc  = w_acc  * torch.exp(-linear_a)
#         reward_jerk = w_jerk * torch.exp(-linear_jerk)
#         reward_snap = w_snap * torch.exp(-linear_snap)


#         r_crash = 1.0 if self._check_done(state) else 0.0

 
#         # 3.5  Final aggregation ----------------------------------------------
#         reward_total = (
#             self.s_angle_diff * angle_diff +
#             self.s_crash * r_crash + 
#             reward_pos +
#             self.s_target  * r_target +
#             # reward_pos * (reward_up + reward_spin) +
#             reward_smooth #+ reward_norm + 
#             # + reward_acc  + reward_jerk   + reward_snap
#         )
#         # print(f"s_angle_diff: {self.s_angle_diff * angle_diff:.4f}")
#         # print(f"s_crash: {self.s_crash * r_crash:.4f}")
#         # print(f"reward_pos: {reward_pos:.4f}")
#         # print(f"s_target: {self.s_target * r_target:.4f}")

#         # 3.5 Aggregate
#         term_pos        = reward_pos
#         term_up         = reward_pos * reward_up
#         term_spin       = reward_pos * reward_spin
#         term_norm       = reward_norm
#         term_smooth     = reward_smooth
#         term_acc        = reward_acc
#         term_jerk       = reward_jerk
#         term_snap       = reward_snap

#         reward = reward_total.item()

#         # print("========== REWARD BREAKDOWN ==========")
#         # print(f"distance          : {distance.item():.4f}")
#         # print(f"term_pos          : {term_pos.item():.4f}")
#         # print(f"term_up           : {term_up.item():.4f}")
#         # print(f"term_spin         : {term_spin.item():.4f}")
#         # print(f"term_norm (w={w_norm:.3f}) : {term_norm.item():.4f}")
#         # print(f"term_smooth       : {term_smooth.item():.4f}")
#         # print(f"term_acc          : {term_acc.item():.4f}")
#         # print(f"term_jerk         : {term_jerk.item():.4f}")
#         # print(f"term_snap         : {term_snap.item():.4f}")
#         # print(f"TOTAL reward      : {reward:.4f}\n")

#         # ---- update history -------------------------------------
#         self.prev_action      = action.clone().detach()
#         self.last_linear_jerk = linear_jerk.item()
#         self.last_linear_a    = linear_a.item()
#         self.last_linear_v    = linear_v.item()
#         self.last_angular_v   = torch.norm(ang_vel).item()
#         self.prev_orientation_q = quat
#         self.prev_position = pos

#         obs_parts = [future_err, rot_mat.flatten(), lin_vel, ang_vel,
# +              torch.tensor(action, device=self.device)]
#         obs_raw = torch.cat(obs_parts)  # (obs_dim,)

#         # normalise to [‑1,1] with predefined scale vector (same as Track)
#         # obs = torch.clamp(obs_raw / torch.tensor(self._obs_scale, device=self.device), -1.0, 1.0).cpu().numpy().astype(torch.float32)
#         # print(f"obs: {obs}")
#         obs = torch.clamp(obs_raw / torch.tensor(self._obs_scale, device=self.device),
#                   -1.0, 1.0) \
#          .cpu() \
#          .numpy() \
#          .astype(np.float32)

#         return obs, reward


#     def _check_done(self,  state: torch.Tensor) -> bool:
#         # pos = torch.array(state[0:3])
#         pos = state[0:3] 
#         t_i = torch.tensor(self._traj_time, device=self.device),
#         dp, _, _ = compute_desired_state(t_i, self.T, self.origin_offset, device=self.device)
#         # pos_error = torch.linalg.norm(dp - pos)
#         if pos[2] < 0.3:
#             return True
#          # 2) 换成“单独三轴误差都不能超过阈值”
#         delta = dp - pos                               # 分量误差 [dx, dy, dz]
#         # 如果任何一个轴的绝对误差超过 self.max_position_error，就 done
#         if torch.any(torch.abs(delta) > self.max_position_error):
#             return True
        
#         qw, qx, qy, qz = state[3:7]
#         roll = torch.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
#         # if abs(roll) > 2.2 or np.linalg.norm(state[7:10]) > self.max_velocity_error:
#         #     return True
#         return False

#     def render(self, mode='human'):
#         pass
    
#     # def quaternion_to_rot_matrix(quat: torch.Tensor,
#     #                          eps: float = 1e-8) -> torch.Tensor:
#     #     # """
#     #     # Convert quaternion (w, x, y, z) to flattened 3x3 rotation matrix.
#     #     # """
#     #     # qw, qx, qy, qz = quat
#     #     # R = torch.array([
#     #     #     [1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
#     #     #     [2*qx*qy + 2*qz*qw,       1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
#     #     #     [2*qx*qz - 2*qy*qw,       2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
#     #     # ], dtype=torch.float32)
#     #     # return R.flatten()
#     #     quat = torch.as_tensor(quat)

#     #     # 拆分分量
#     #     w, x, y, z = quat.unbind(-1)

#     #     # 归一化（保证数值稳定）
#     #     norm = torch.sqrt(w*w + x*x + y*y + z*z + eps)
#     #     w, x, y, z = w/norm, x/norm, y/norm, z/norm

#     #     # 旋转矩阵各元素
#     #     ww, xx, yy, zz = w*w, x*x, y*y, z*z
#     #     xy, xz, yz     = x*y, x*z, y*z
#     #     wx, wy, wz     = w*x, w*y, w*z

#     #     R00 = 1 - 2*(yy + zz)
#     #     R01 =     2*(xy - wz)
#     #     R02 =     2*(xz + wy)

#     #     R10 =     2*(xy + wz)
#     #     R11 = 1 - 2*(xx + zz)
#     #     R12 =     2*(yz - wx)

#     #     R20 =     2*(xz - wy)
#     #     R21 =     2*(yz + wx)
#     #     R22 = 1 - 2*(xx + yy)

#     #     R = torch.stack([
#     #         torch.stack([R00, R01, R02], dim=-1),
#     #         torch.stack([R10, R11, R12], dim=-1),
#     #         torch.stack([R20, R21, R22], dim=-1)
#     #     ], dim=-2)           # (..., 3, 3)

#     #     return R
    
#     def close(self):
#         rospy.signal_shutdown("Environment closed")

































# # --------------------------------------------------




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


        self.FUTURE_STEPS = 10

        # ---- Origin offset (multi‑drone support) --------------------
        self.origin_offset = -5.0
        for i in range(10):
            if str(i) in namespace:
                self.origin_offset += 3.0 * i

        # ---- Episode bookkeeping -----------------------------------
        self.max_episode_steps = 256
        self.step_count = 0
        self.episode_reward = 0.0

        # ---- Reward weights ----------------------------------------
        self.s_angle_diff = -1.0
        self.s_target = 0.5
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
        self.reward_action_smoothness_weight_init= 0.4
        self.reward_action_smoothness_weight_lr= 0.0005 # slow= 0.0001, fast= 0.0005
        self.reward_smoothness_max= 1.0
        # action norm
        self.reward_action_norm_weight_init= 0.0
        self.reward_action_norm_weight_lr= 0.0001 # slow= 0.0001, fast= 0.0005
        self.reward_norm_max= 0.0
        # distance
        self.reward_distance_scale= 5.0
        self.s_crash = -10

        # ---- Dynamics history --------------------------------------
        self.action_dim = 4
        self.prev_action = np.zeros(self.action_dim, np.float32)
        self.last_linear_v = 0.0
        self.last_angular_v = 0.0
        self.last_linear_a = 0.0
        self.last_angular_a = 0.0
        self.last_linear_jerk = 0.0
        # self.prev_orientation_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        # self.prev_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        # store history as NumPy arrays instead of torch.Tensors
        self.prev_orientation_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_position      = np.array([0.0, 0.0, 0.0],       dtype=np.float32)

        # ---- Safety bounds (also used for scaling) -----------------
        self.max_position_error  = 2.5      # m
        self.max_velocity_error  = 6.0      # m/s

        # ---- ROS ----------------------------------------------------
        if not rospy.core.is_initialized():
            rospy.set_param('use_sim_time', True)
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
        self.T            = 10           # sec per loop
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
        dt = (now - self._last_step_time).to_sec()
        # print(f"dt: {dt}")
        self._last_step_time = now
        self._traj_time += dt
        self._publish_desired_marker()

        cmd = ControlCommand()
        cmd.armed = True;  cmd.control_mode = ControlCommand.BODY_RATES
        cmd.collective_thrust = ((action[0] + 1.0) / 2.0) * (self.max_thrust - self.min_thrust) + self.min_thrust
        cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = action[1:] * self.max_angular_rate
        self.cmd_pub.publish(cmd)
        # print(f"Rospy.time: {rospy.Time.now().to_sec()}")
        rospy.wait_for_message(f"/{self.namespace}/ground_truth/odometry", Odometry)
   
        # print(f"Rospy.time: {rospy.Time.now().to_sec()}")
        # ----- observe --------------------------------------------
        with self._odom_lock:
            state = self.current_state.copy()
 
        obs, reward = self._build_obs_and_reward(state, action)
 
        # ----- update episode bookkeeping ------------------------
        self.step_count += 1
        self.episode_reward += reward

        # print(f"step: {self.step_count}, self.episode_reward: {self.episode_reward}")
        done = self._check_done(state) or self.step_count >= self.max_episode_steps

        info = {"reward": reward}
        # print(f"step: {self.step_count}, reward: {reward:.4f}, done: {done}")
        if done:
            info["episode"] = {"r": self.episode_reward, "l": self.step_count}
            self.step_count = 0; self.episode_reward = 0.0
        return obs, reward, done, False, info

    def reset(self, **kwargs):
        self.step_count = 0; 
        self.episode_reward = 0.0; 
        self.prev_action.fill(0.0)
        self._traj_time = 0.0; 
        self._last_step_time = rospy.Time.now()
        self.prev_orientation_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_position = np.array([ self.origin_offset + 1, 0, 1], dtype=np.float32)
        self.T = random.randint(3, 15)

        # place drone near origin offset
        reset_pos = np.array([
            self.origin_offset + np.random.uniform(-1, 1),
            np.random.uniform(-1, 1), np.random.uniform(0.6, 1.8)
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
            m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = np.random.uniform(-0.2, 0.2, 3)
            m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = np.random.uniform(-0.1, 0.1, 3)
            pub.publish(m)
            with self._odom_lock:
                if np.linalg.norm(self.current_state[0:3] - p) < 0.1 or rospy.Time.now() - start > timeout:
                    break

    def safe_normalize(self, q, epsilon=1e-6):
        """
        安全归一化四元数，防止除以零
        """
        q = np.array(q)  # 转换为 NumPy 数组
        norm = np.linalg.norm(q)
        if norm < epsilon:
            norm = epsilon
        return q / norm
    
    

    # ----------------------- Observation & reward ------------------
    def _build_obs_and_reward(self, state: np.ndarray, action: np.ndarray):
        device = self.device
        # ---------------------------------------------------------------------
        # 1.  Tensor helpers (pos / vel / rot)
        # ---------------------------------------------------------------------
        pos     = np.array(state[0:3])
        quat    = np.array(state[3:7])
        lin_vel = np.array(state[7:10])
        ang_vel = np.array(state[10:13])

         # rotation matrix, up‑vector
        rot_flat = self.quaternion_to_rot_matrix(quat)     # returns np.ndarray of shape (9,)
        rot_mat  = rot_flat.reshape(3, 3)
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
            desired_pos.append(p_i)
            # desired_pos.append(torch.tensor(p_i, device=self.device))
        # desired_pos = torch.stack(desired_pos)                                # (K,3)
        future_err  = (desired_pos - pos).flatten()                           # (3K,)
        # print(f"future_err: {future_err}")
        # ---------------------------------------------------------------------
        # 3.  Rewards  ---------------------------------------------------------
        # ---------------------------------------------------------------------
        # 3.1  Main distance term (only the 0‑th waypoint)
        # distance        = torch.norm(future_err[0:3])
        curr_rel_dis = np.clip(np.linalg.norm(future_err[0:3])/self.max_position_error,0,1)
        # curr_rel_dis = torch.clamp(np.linalg.norm(future_err[0:3])/self.max_position_error, min=0.0, max=1.0)
        r_position = np.exp(-1.0 * curr_rel_dis)
        reward_pos = self.reward_distance_scale * r_position
        # print(f"reward_pos: {reward_pos.item():.4f}")


        prev_dist = np.linalg.norm(self.prev_position - desired_pos[0:3])
        curr_dist = np.linalg.norm(pos - desired_pos[0:3])
        
        r_progress_dis = prev_dist - curr_dist
        
        r_target = np.tanh(30*r_progress_dis)
        # print(f"r_target: {r_target:.4f}")


        # 3. 惩罚无人机的姿态变化
        epsilon=1e-6
        q_prev = self.safe_normalize(self.prev_orientation_q, epsilon)
        q_curr = self.safe_normalize(quat, epsilon)

        # 计算内积并裁剪到[-1, 1]范围内
        dot_product = np.dot(q_prev, q_curr)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # 计算旋转角度差
        angle_diff = 2 * np.arccos(abs(dot_product))


        # 3.2  Uprightness & spin  (gated by reward_pos later)
        tiltage         = np.abs(1.0 - up_z)
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
            delta_a_norm  = torch.norm(delta_a)/5.0
            # w_smooth = min(self.reward_action_smoothness_weight_init +
            #                 self.reward_action_smoothness_weight_lr * self.step_count,
            #                 self.reward_smoothness_max)
 
            reward_smooth = torch.clamp(-0.5 * delta_a_norm, min=-1.0, max=1.0)
            # print(f"reward_smooth: {reward_smooth.item():.4f}")

        # 3.4  Dynamics regularisation (acc / jerk / snap) --------------------
        # linear_v     = torch.norm(lin_vel)
        # if self.step_count < 2:                 # 先收两帧再算
        #     linear_a = torch.tensor(0., device=self.device)
        #     linear_jerk = torch.tensor(0., device=self.device)
        #     linear_snap = torch.tensor(0., device=self.device)
        # else:
        #     linear_v = torch.norm(lin_vel)
        #     linear_a = torch.abs(linear_v - self.last_linear_v) / self.control_dt
        #     linear_jerk = torch.abs(linear_a - self.last_linear_a) / self.control_dt
        #     linear_snap = torch.abs(linear_jerk - self.last_linear_jerk) / self.control_dt

 
        # w_acc  = min(self.reward_acc_weight_init  + self.reward_acc_weight_lr  * self.step_count, self.reward_acc_max)
        # w_jerk = min(self.reward_jerk_weight_init + self.reward_jerk_weight_lr * self.step_count, self.reward_jerk_max)
        # w_snap = min(self.reward_snap_weight_init + self.reward_snap_weight_lr * self.step_count, self.reward_snap_max)

        # reward_acc  = w_acc  * torch.exp(-linear_a)
        # reward_jerk = w_jerk * torch.exp(-linear_jerk)
        # reward_snap = w_snap * torch.exp(-linear_snap)


        r_crash = 1.0 if self._check_done(state) else 0.0

 
        # 3.5  Final aggregation ----------------------------------------------
        reward_total = (
            self.s_angle_diff * angle_diff +
            self.s_crash * r_crash + 
            reward_pos +
            self.s_target  * r_target +
            # reward_pos * (reward_up + reward_spin) +
            reward_smooth #+ reward_norm + 
            # + reward_acc  + reward_jerk   + reward_snap
        )
        # print(f"s_angle_diff: {self.s_angle_diff * angle_diff:.4f}")
        # print(f"s_crash: {self.s_crash * r_crash:.4f}")
        # print(f"reward_pos: {reward_pos:.4f}")
        # print(f"s_target: {self.s_target * r_target:.4f}")

        # 3.5 Aggregate
        term_pos        = reward_pos
        term_up         = reward_pos * reward_up
        term_spin       = reward_pos * reward_spin
        term_norm       = reward_norm
        term_smooth     = reward_smooth


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
        # self.last_linear_jerk = linear_jerk.item()
        # self.last_linear_a    = linear_a.item()
        # self.last_linear_v    = linear_v.item()
        self.last_angular_v   = ang_vel 
        self.prev_orientation_q = quat
        self.prev_position = pos

        obs_raw = np.concatenate([future_err, rot_mat.flatten(), lin_vel, ang_vel,
              np.array(action, dtype=np.float32)])
        obs = np.clip(obs_raw / self._obs_scale, -1.0, 1.0)
        # print(f"obs: {obs}")

        return obs, reward


    def _check_done(self, state):
        pos = np.array(state[0:3])
        t_i = self._traj_time
        dp, _, _ = compute_desired_state(t_i, self.T, self.origin_offset)
        pos_error = np.linalg.norm(dp - pos)
        if pos[2] < 0.3:
            return True
         # 2) 换成“单独三轴误差都不能超过阈值”
        delta = dp - pos                               # 分量误差 [dx, dy, dz]
        # 如果任何一个轴的绝对误差超过 self.max_position_error，就 done
        if np.any(np.abs(delta) > self.max_position_error):
            return True
        
        qw, qx, qy, qz = state[3:7]
        roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
        # if abs(roll) > 2.2 or np.linalg.norm(state[7:10]) > self.max_velocity_error:
        #     return True
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


