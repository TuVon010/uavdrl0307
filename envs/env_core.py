import numpy as np
from envs.Base import Base
from envs.physics_engine import PhysicsEngine


class EnvCore(object):
    """
    UAV辅助医疗IoT任务卸载多智能体环境
    智能体: n_users个用户 + n_uavs个无人机
    用户动作: [卸载比例, 关联服务器logits(local + M个UAV)]
    无人机动作: [速度, 方向, 对每个用户的算力分配]
    """

    def __init__(self):
        self.base = Base()
        self.physics = PhysicsEngine(self.base)

        self.n_users = self.base.n_users
        self.n_uavs = self.base.n_uavs
        self.agent_num = self.n_users + self.n_uavs

        # 统一观测维度 (所有智能体相同, 便于框架处理)
        # all_user_pos(2*U) + all_uav_pos(2*M) + own_energy(1)
        # + all_task_sizes(U) + all_task_cycles(U) + agent_type(1) + agent_id(1)
        self.obs_dim = (2 * self.n_users + 2 * self.n_uavs
                        + 1 + self.n_users + self.n_users + 1 + 1)

        # 异构动作维度
        # 用户: ratio(1) + assoc_logits(1+M) = 1 + 1 + n_uavs
        self.user_action_dim = 1 + 1 + self.n_uavs
        # 无人机: speed(1) + direction(1) + freq_per_user(U) = 2 + n_users
        self.uav_action_dim = 2 + self.n_users
        self.action_dims = ([self.user_action_dim] * self.n_users
                            + [self.uav_action_dim] * self.n_uavs)

        self.max_steps = 60
        self.current_step = 0

        self.users = None
        self.uavs = None
        self.tasks = None

    # =========================================================
    # 初始化
    # =========================================================
    def reset(self):
        self.current_step = 0
        self._init_users()
        self._init_uavs()
        self._generate_tasks()
        return self._get_obs()

    def _init_users(self):
        self.users = []
        cx = (self.base.field_X[0] + self.base.field_X[1]) / 2
        cy = (self.base.field_Y[0] + self.base.field_Y[1]) / 2
        spread = self.base.user_cluster_radius
        for _ in range(self.n_users):
            x = np.clip(np.random.normal(cx, spread),
                        self.base.field_X[0], self.base.field_X[1])
            y = np.clip(np.random.normal(cy, spread),
                        self.base.field_Y[0], self.base.field_Y[1])
            self.users.append({
                'position': np.array([x, y]),
                'velocity': np.random.uniform(0.0, self.base.user_mean_velocity * 2),
                'direction': np.random.uniform(0, 2 * np.pi),
                'energy': 0.0,
                'priority': np.random.choice([0, 1], p=[0.8, 0.2]),
            })

    def _init_uavs(self):
        self.uavs = []
        cx = (self.base.field_X[0] + self.base.field_X[1]) / 2
        cy = (self.base.field_Y[0] + self.base.field_Y[1]) / 2
        r = self.base.uav_init_radius
        for j in range(self.n_uavs):
            angle = 2 * np.pi * j / self.n_uavs + np.pi / 2
            ux = np.clip(cx + r * np.cos(angle),
                         self.base.field_X[0], self.base.field_X[1])
            uy = np.clip(cy + r * np.sin(angle),
                         self.base.field_Y[0], self.base.field_Y[1])
            self.uavs.append({
                'position': np.array([ux, uy]),
                'energy': 0.0,
                'cumulative_energy': 0.0,
            })

    def _generate_tasks(self):
        self.tasks = []
        for _ in range(self.n_users):
            self.tasks.append({
                'data_size': np.random.uniform(
                    self.base.task_size_min, self.base.task_size_max),
                'cpu_cycles': np.random.uniform(
                    self.base.cycles_min, self.base.cycles_max),
                'deadline': self.base.latency_max,
            })

    # =========================================================
    # 观测构建
    # =========================================================
    def _get_obs(self):
        user_pos = np.array([u['position'] for u in self.users]).flatten() / self.base.norm_pos
        uav_pos = np.array([u['position'] for u in self.uavs]).flatten() / self.base.norm_pos
        task_sizes = np.array([t['data_size'] for t in self.tasks]) / self.base.norm_data
        task_cycles = np.array([t['cpu_cycles'] for t in self.tasks]) / self.base.norm_cycle

        obs_list = []
        for i in range(self.n_users):
            obs = np.concatenate([
                user_pos,
                uav_pos,
                [self.users[i]['energy'] / self.base.norm_energy_user],
                task_sizes,
                task_cycles,
                [0.0],
                [i / self.agent_num],
            ]).astype(np.float32)
            obs_list.append(obs)

        for j in range(self.n_uavs):
            obs = np.concatenate([
                user_pos,
                uav_pos,
                [self.uavs[j]['energy'] / self.base.norm_energy_uav],
                task_sizes,
                task_cycles,
                [1.0],
                [(self.n_users + j) / self.agent_num],
            ]).astype(np.float32)
            obs_list.append(obs)

        return obs_list

    # =========================================================
    # 环境步进
    # =========================================================
    def step(self, actions):
        # ---------- 1. 解析动作 ----------
        user_ratios, user_assocs = self._parse_user_actions(actions)
        uav_speeds, uav_dirs, uav_freqs = self._parse_uav_actions(actions)

        # ---------- 2. 更新位置 ----------
        self._update_uav_positions(uav_speeds, uav_dirs)
        self.physics.update_user_positions(self.users)

        # ---------- 3. 统计每个UAV关联的用户 ----------
        users_per_uav = [[] for _ in range(self.n_uavs)]
        for i in range(self.n_users):
            if user_assocs[i] > 0:
                users_per_uav[user_assocs[i] - 1].append(i)

        # ---------- 4. 计算时延和能耗 ----------
        (user_delays, user_energies, uav_comp_energies,
         uav_fly_energies, deadline_violations) = self._compute_delays_and_energies(
            user_ratios, user_assocs, uav_freqs, uav_speeds, users_per_uav)

        # ---------- 5. 计算奖励 ----------
        rewards, reward_details = self._compute_rewards(
            user_delays, user_energies, deadline_violations,
            uav_fly_energies, uav_comp_energies, users_per_uav)

        # ---------- 6. 新任务 + 新观测 ----------
        self._generate_tasks()
        self.current_step += 1
        obs_list = self._get_obs()

        done = self.current_step >= self.max_steps
        dones = [done] * self.agent_num

        infos = self._build_infos(
            user_delays, user_energies, deadline_violations,
            uav_fly_energies, uav_comp_energies, reward_details)

        return [obs_list, rewards, dones, infos]

    # =========================================================
    # 动作解析
    # =========================================================
    def _parse_user_actions(self, actions):
        user_ratios = []
        user_assocs = []
        for i in range(self.n_users):
            act = actions[i]
            ratio = _sigmoid(act[0])
            assoc = int(np.argmax(act[1:1 + 1 + self.n_uavs]))
            if assoc == 0:
                ratio = 0.0
            user_ratios.append(ratio)
            user_assocs.append(assoc)
        return user_ratios, user_assocs

    def _parse_uav_actions(self, actions):
        uav_speeds = []
        uav_dirs = []
        uav_freqs = []
        for j in range(self.n_uavs):
            act = actions[self.n_users + j]
            speed = _sigmoid(act[0]) * self.base.uav_v_max
            direction = np.tanh(act[1]) * np.pi
            raw_f = act[2:2 + self.n_users]
            exp_f = np.exp(raw_f - np.max(raw_f))
            freq = (exp_f / np.sum(exp_f)) * self.base.C_uav
            uav_speeds.append(speed)
            uav_dirs.append(direction)
            uav_freqs.append(freq)
        return uav_speeds, uav_dirs, uav_freqs

    # =========================================================
    # 位置更新
    # =========================================================
    def _update_uav_positions(self, uav_speeds, uav_dirs):
        for j in range(self.n_uavs):
            dx = uav_speeds[j] * np.cos(uav_dirs[j]) * self.base.time_step
            dy = uav_speeds[j] * np.sin(uav_dirs[j]) * self.base.time_step
            new_pos = self.uavs[j]['position'] + np.array([dx, dy])
            new_pos[0] = np.clip(new_pos[0], self.base.field_X[0], self.base.field_X[1])
            new_pos[1] = np.clip(new_pos[1], self.base.field_Y[0], self.base.field_Y[1])
            self.uavs[j]['position'] = new_pos

    # =========================================================
    # 时延 & 能耗计算 (论文公式 14-24)
    # =========================================================
    def _compute_delays_and_energies(self, user_ratios, user_assocs,
                                     uav_freqs, uav_speeds, users_per_uav):
        user_delays = np.zeros(self.n_users)
        user_energies = np.zeros(self.n_users)
        uav_comp_energies = np.zeros(self.n_uavs)
        uav_fly_energies = np.zeros(self.n_uavs)
        deadline_violations = np.zeros(self.n_users)

        for i in range(self.n_users):
            D = self.tasks[i]['data_size']
            C = self.tasks[i]['cpu_cycles']
            tau = self.tasks[i]['deadline']
            lam = user_ratios[i]
            assoc = user_assocs[i]
            local_frac = 1.0 - lam

            # --- 本地计算 (eq.14, eq.19) ---
            T_local = 0.0
            E_local = 0.0
            if local_frac > 1e-8:
                T_local = local_frac * D * C / self.base.C_local
                E_local = self.base.k_local * (self.base.C_local ** 2) * local_frac * D * C

            # --- 卸载计算 (eq.15, eq.20) ---
            T_off = 0.0
            E_tx = 0.0
            if assoc > 0 and lam > 1e-8:
                uav_idx = assoc - 1
                g = self.physics.get_channel_gain(
                    self.users[i]['position'], self.uavs[uav_idx]['position'])
                n_assoc = max(len(users_per_uav[uav_idx]), 1)
                bw = self.base.B_total / n_assoc
                R = self.physics.compute_rate(g, bw, self.base.p_tx_max)

                if R > 1e-3:
                    T_tx = lam * D / R
                else:
                    T_tx = tau * 100.0

                f_mu = max(uav_freqs[uav_idx][i], 1e3)
                T_comp_uav = lam * D * C / f_mu
                T_off = T_tx + T_comp_uav

                E_tx = self.base.p_tx_max * T_tx
                uav_comp_energies[uav_idx] += self.base.xi_m * lam * D * C

            # --- 任务完成时延 (eq.17) ---
            T_u = max(T_local, T_off)
            T_u = min(T_u, tau * 10.0)  # cap to avoid unbounded values
            user_delays[i] = T_u

            # --- 用户总能耗 (eq.21) ---
            E_u = E_local + E_tx
            user_energies[i] = E_u
            self.users[i]['energy'] = E_u

            if T_u > tau:
                deadline_violations[i] = 1.0

        # --- 无人机飞行能耗 (eq.22, eq.24) ---
        for j in range(self.n_uavs):
            E_fly = self.physics.compute_uav_energy(uav_speeds[j])
            uav_fly_energies[j] = E_fly
            E_total = E_fly + uav_comp_energies[j]
            self.uavs[j]['energy'] = E_total
            self.uavs[j]['cumulative_energy'] += E_total

        return (user_delays, user_energies, uav_comp_energies,
                uav_fly_energies, deadline_violations)

    # =========================================================
    # 奖励计算
    # =========================================================
    def _compute_rewards(self, user_delays, user_energies, violations,
                         uav_fly_e, uav_comp_e, users_per_uav):
        b = self.base

        # ---- 系统成本 (eq.26-28, 归一化) ----
        delay_costs = np.zeros(self.n_users)
        for i in range(self.n_users):
            omega = b.omega_H if self.users[i]['priority'] == 1 else b.omega_L
            delay_costs[i] = omega * user_delays[i] / b.latency_max

        avg_delay_cost = np.mean(delay_costs)

        avg_energy_cost = (
            np.sum(user_energies / b.norm_energy_user) / self.n_users
            + np.sum((uav_fly_e + uav_comp_e) / b.norm_energy_uav) / self.n_uavs
        ) / 2.0

        total_cost = b.mu_L * avg_delay_cost + b.mu_E * avg_energy_cost
        penalty = -b.w_penalty * np.mean(violations)
        system_reward = -total_cost + penalty

        rewards = []
        reward_details = []

        # ---- 用户奖励: w1*系统奖励 - w2*用户成本 ----
        user_rewards_raw = []
        for i in range(self.n_users):
            omega = b.omega_H if self.users[i]['priority'] == 1 else b.omega_L
            delay_ratio = min(user_delays[i] / b.latency_max, 10.0)
            energy_ratio = min(user_energies[i] / b.norm_energy_user, 5.0)
            user_cost = omega * (delay_ratio + energy_ratio)
            w1_sys = b.w1_user * system_reward
            neg_w2_cost = -b.w2_user * user_cost
            r = np.clip(w1_sys + neg_w2_cost, -10.0, 10.0)
            user_rewards_raw.append(r)
            rewards.append([r])
            reward_details.append({
                'agent_type': 'user',
                'system_reward': float(system_reward),
                'neg_total_cost': float(-total_cost),
                'penalty': float(penalty),
                'w1_system': float(w1_sys),
                'user_cost': float(user_cost),
                'neg_w2_cost': float(neg_w2_cost),
                'total': float(r),
            })

        # ---- 无人机奖励: w_sys*系统奖励 + w_ind*无人机个体奖励 ----
        for j in range(self.n_uavs):
            pos = self.uavs[j]['position']

            boundary_pen = 0.0
            dist_to_edge = min(
                pos[0] - b.field_X[0], b.field_X[1] - pos[0],
                pos[1] - b.field_Y[0], b.field_Y[1] - pos[1])
            if dist_to_edge < b.boundary_warn:
                boundary_pen = b.w_guide * (b.boundary_warn - dist_to_edge) / b.boundary_warn

            collision_pen = 0.0
            for k in range(self.n_uavs):
                if k != j:
                    d = np.linalg.norm(pos - self.uavs[k]['position'])
                    if d < b.uav_safe_dist:
                        collision_pen += b.w_collision * (b.uav_safe_dist - d) / b.uav_safe_dist

            assoc_sum = sum(user_rewards_raw[uid] for uid in users_per_uav[j]) if users_per_uav[j] else 0.0
            n_assoc = max(len(users_per_uav[j]), 1)
            assoc_avg = assoc_sum / n_assoc
            uav_individual = assoc_avg - boundary_pen - collision_pen

            w_sys_part = b.w_sys_uav * system_reward
            w_ind_part = b.w_ind_uav * uav_individual
            r_uav = np.clip(w_sys_part + w_ind_part, -10.0, 10.0)
            rewards.append([r_uav])
            reward_details.append({
                'agent_type': 'uav',
                'system_reward': float(system_reward),
                'neg_total_cost': float(-total_cost),
                'penalty': float(penalty),
                'w_sys_part': float(w_sys_part),
                'assoc_reward_avg': float(assoc_avg),
                'boundary_pen': float(boundary_pen),
                'collision_pen': float(collision_pen),
                'uav_individual': float(uav_individual),
                'w_ind_part': float(w_ind_part),
                'total': float(r_uav),
            })

        return rewards, reward_details

    # =========================================================
    # 辅助
    # =========================================================
    def _build_infos(self, user_delays, user_energies, violations,
                     uav_fly_e, uav_comp_e, reward_details):
        total_sys_energy = float(np.sum(user_energies) + np.sum(uav_fly_e + uav_comp_e))
        avg_delay = float(np.mean(user_delays))

        infos = []
        for i in range(self.n_users):
            infos.append({
                'delay': float(user_delays[i]),
                'energy': float(user_energies[i]),
                'violation': float(violations[i]),
                'position': self.users[i]['position'].copy(),
                'reward_details': reward_details[i],
                'total_system_energy': total_sys_energy,
                'avg_user_delay': avg_delay,
            })
        for j in range(self.n_uavs):
            infos.append({
                'fly_energy': float(uav_fly_e[j]),
                'comp_energy': float(uav_comp_e[j]),
                'cumulative_energy': float(self.uavs[j]['cumulative_energy']),
                'position': self.uavs[j]['position'].copy(),
                'reward_details': reward_details[self.n_users + j],
                'total_system_energy': total_sys_energy,
                'avg_user_delay': avg_delay,
            })
        return infos


def _sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))
