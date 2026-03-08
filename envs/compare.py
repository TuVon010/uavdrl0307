def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            # 基本位置与系统指标记录
            ep_user_pos = [[] for _ in range(self.n_users)]
            ep_uav_pos = [[] for _ in range(self.n_uavs)]
            ep_delays = []
            ep_sys_energies = []
            
            # ==========================================
            # 1. 初始化【奖励详情表】的容器和累计变量
            # ==========================================
            ep_reward_rows = []
            sys_cum_reward = 0.0 # 系统的累计大锅饭奖励
            agent_cum_rewards = np.zeros(self.num_agents) # 每个智能体的累计总分
            
            # ==========================================
            # 2. 初始化【3张性能指标表】的容器和累计变量
            # ==========================================
            ep_user_metrics = []
            ep_uav_metrics = []
            ep_sys_metrics = []
            user_cum_energy = np.zeros(self.n_users)
            sys_cum_time_cost = 0.0
            sys_cum_energy = 0.0

            for step in range(self.episode_length):
                (
                    values, actions, action_log_probs,
                    rnn_states, rnn_states_critic, actions_env,
                ) = self.collect(step)

                obs, rewards, dones, infos = self.envs.step(actions_env)

                # --- 从第一个环境线程收集指标 ---
                info_e0 = infos[0]
                
                # 记录位置轨迹和基础指标
                for i in range(self.n_users):
                    ep_user_pos[i].append(info_e0[i]['position'].copy())
                for j in range(self.n_uavs):
                    ep_uav_pos[j].append(info_e0[self.n_users + j]['position'].copy())
                ep_delays.append(info_e0[0]['avg_user_delay'])
                ep_sys_energies.append(info_e0[0]['total_system_energy'])

                # ========================================================
                # 记录 A：提取并保存 3张【性能指标表】需要的数据
                # ========================================================
                if episode % 10 == 0:
                    sys_cum_time_cost, sys_cum_energy = self._collect_performance_metrics(
                        step, info_e0,
                        ep_sys_metrics, ep_user_metrics, ep_uav_metrics,
                        sys_cum_time_cost, sys_cum_energy, user_cum_energy
                    )

                # ========================================================
                # 记录 B：提取并保存高度拆解的【奖励详情表】数据
                # ========================================================
                if episode % 10 == 0:
                    # 1. 提取本步的系统大锅饭奖励 (全局共享，直接取第0个就行)
                    step_sys_reward = info_e0[0]['reward_details']['system_reward']
                    sys_cum_reward += step_sys_reward

                    # 2. 遍历每个智能体，分类提取数据
                    for a in range(self.num_agents):
                        rd = info_e0[a]['reward_details']
                        step_total = rd['total']
                        agent_cum_rewards[a] += step_total
                        
                        # 生成当前步的报表行 (利用 .get() 方法，没有的项自动补 0.0)
                        row = {
                            'Step': step,
                            'Agent_Type': rd['agent_type'].upper(), # 'USER' 或 'UAV'
                            'Agent_ID': a if a < self.n_users else a - self.n_users,
                            
                            # --- 全局共有项 ---
                            'System_Reward': step_sys_reward,
                            'System_Cumulative_Reward': sys_cum_reward,
                            'Agent_Step_Total_Reward': step_total,
                            'Agent_Cumulative_Reward': agent_cum_rewards[a],
                            
                            # --- 用户专属字段 (UAV填0) ---
                            'w1_System_Reward': rd.get('w1_system', 0.0),
                            'User_Cost_Penalty': rd.get('neg_w2_cost', 0.0),
                            'Norm_Delay_Ratio': rd.get('delay_ratio', 0.0),
                            'Norm_Energy_Ratio': rd.get('energy_ratio', 0.0),
                            
                            # --- 无人机专属字段 (User填0) ---
                            'w_Sys_Part_Reward': rd.get('w_sys_part', 0.0),
                            'Proximity_Reward': rd.get('proximity_reward', 0.0),
                            'Coverage_Reward': rd.get('coverage_reward', 0.0),
                            'Association_Bonus': rd.get('assoc_bonus', 0.0),
                            'Boundary_Penalty': rd.get('boundary_pen', 0.0),
                            'Collision_Penalty': rd.get('collision_pen', 0.0)
                        }
                        ep_reward_rows.append(row)

                # Buffer 插入和网络更新
                data = (obs, rewards, dones, infos, values, actions,
                        action_log_probs, rnn_states, rnn_states_critic)
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # --- 每个轮次: 保存轨迹图 ---
            self._save_trajectory(episode, ep_user_pos, ep_uav_pos)

            # --- 每5个轮次: 打印平均时延和平均能耗 ---
            if episode % 5 == 0:
                self._print_episode_metrics(episode, ep_delays, ep_sys_energies)

            # --- 每10个轮次: 保存所有的表格 ---
            if episode % 10 == 0:
                self._save_reward_table(episode, ep_reward_rows)
                self._save_performance_metrics(episode, ep_user_metrics, ep_uav_metrics, ep_sys_metrics)

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information (原有代码保持不变)
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                        self.all_args.scenario_name, self.algorithm_name, self.experiment_name, episode, episodes,
                        total_num_steps, self.num_env_steps, int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    pass # 根据你原代码，这里有 MPE 的逻辑
                else:
                    for agent_id in range(self.num_agents):
                        avg_rew = np.mean(self.buffer[agent_id].rewards) * self.episode_length
                        train_infos[agent_id].update({"average_episode_rewards": avg_rew})
                    avg_all = np.mean([np.mean(self.buffer[a].rewards) for a in range(self.num_agents)])
                    print("  average reward: {:.4f}".format(avg_all * self.episode_length))

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    # =================================================================
    # 数据收集与表格保存方法 (直接跟在 run 后面)
    # =================================================================
    def _collect_performance_metrics(self, step, info_e0, 
                                     ep_sys_metrics, ep_user_metrics, ep_uav_metrics,
                                     sys_cum_time_cost, sys_cum_energy, user_cum_energy):
        """提取并保存 3 张性能表 (系统、用户、无人机)"""
        # 1. 记录系统级数据
        sys_step_time = info_e0[0].get('sys_time_cost', 0.0)
        sys_step_eng = info_e0[0]['total_system_energy']
        sys_cum_time_cost += sys_step_time
        sys_cum_energy += sys_step_eng
        
        w_L = self.base_config.mu_L  # 确保你的 __init__ 里实例化了 self.base_config = Base()
        w_E = self.base_config.mu_E
        obj_val = w_L * sys_step_time + w_E * sys_step_eng
        
        ep_sys_metrics.append({
            'Step': step,
            'Sys_Time_Cost': sys_step_time,
            'Sys_Energy_J': sys_step_eng,
            'Cum_Sys_Time_Cost': sys_cum_time_cost,
            'Cum_Sys_Energy_J': sys_cum_energy,
            'Objective_Value_Cost': obj_val,
            'mu_L_Weight': w_L,
            'mu_E_Weight': w_E
        })

        # 2. 记录 User 级数据
        for i in range(self.n_users):
            eng = info_e0[i]['energy']
            user_cum_energy[i] += eng
            assoc_id = info_e0[i].get('association', 0)
            assoc_str = "Local" if assoc_id == 0 else f"UAV_{assoc_id-1}"
            
            ep_user_metrics.append({
                'Step': step,
                'User_ID': i,
                'Latency_s': info_e0[i]['delay'],
                'Association': assoc_str,
                'Offload_Ratio': info_e0[i].get('offload_ratio', 0.0),
                'Allocated_Freq_Hz': info_e0[i].get('alloc_freq', 0.0), 
                'Energy_J': eng,
                'Cum_Energy_J': user_cum_energy[i]
            })

        # 3. 记录 UAV 级数据
        for j in range(self.n_uavs):
            idx = self.n_users + j
            f_eng = info_e0[idx]['fly_energy']
            c_eng = info_e0[idx]['comp_energy']
            ep_uav_metrics.append({
                'Step': step,
                'UAV_ID': j,
                'Pos_X': info_e0[idx]['position'][0],
                'Pos_Y': info_e0[idx]['position'][1],
                'Fly_Energy_J': f_eng,
                'Comp_Energy_J': c_eng,
                'Total_Energy_J': f_eng + c_eng,
                'Cum_Energy_J': info_e0[idx]['cumulative_energy']
            })
            
        return sys_cum_time_cost, sys_cum_energy

    def _save_performance_metrics(self, episode, user_rows, uav_rows, sys_rows):
        """把性能指标保存为独立的 CSV"""
        if not user_rows: return
        pd.DataFrame(user_rows).to_csv(os.path.join(self.metrics_dir, f'ep_{episode}_user_metrics.csv'), index=False)
        pd.DataFrame(uav_rows).to_csv(os.path.join(self.metrics_dir, f'ep_{episode}_uav_metrics.csv'), index=False)
        pd.DataFrame(sys_rows).to_csv(os.path.join(self.metrics_dir, f'ep_{episode}_system_metrics.csv'), index=False)
        print(f"  >>> Saved 3 Performance Tables (User, UAV, Sys) to {self.metrics_dir}")

    def _save_reward_table(self, episode, ep_reward_rows):
        """保存高度拆解的奖励详情表 (带中文公式表头)"""
        path = os.path.join(self.reward_dir, f'episode_{episode}_rewards.csv')
        if not ep_reward_rows:
            return

        # 1. 第一行的中文公式说明
        formula_text = (
            "说明：用户奖励 = 0.4 * 系统奖励 - 0.6 * 个体成本惩罚 (其中 个体成本 = 加权归一化时延 + 加权归一化能耗) | "
            "无人机奖励 = 0.3 * 系统奖励 + 0.7 * (接近中心 + 覆盖 + 关联接客 - 越界惩罚 - 碰撞惩罚)\n"
        )

        # 2. 转为 pandas DataFrame 并控制列的先后顺序
        df = pd.DataFrame(ep_reward_rows)
        cols_order = [
            'Step', 'Agent_Type', 'Agent_ID', 
            'System_Reward', 'System_Cumulative_Reward', 
            'Agent_Step_Total_Reward', 'Agent_Cumulative_Reward',
            'w1_System_Reward', 'User_Cost_Penalty', 'Norm_Delay_Ratio', 'Norm_Energy_Ratio', 
            'w_Sys_Part_Reward', 'Proximity_Reward', 'Coverage_Reward', 'Association_Bonus', 'Boundary_Penalty', 'Collision_Penalty'
        ]
        df = df[cols_order]

        # 3. 写入文件 (使用 utf-8-sig 确保 Excel 打开中文不乱码)
        with open(path, 'w', encoding='utf-8-sig') as f:
            f.write(formula_text)         
            df.to_csv(f, index=False)
        print(f"  >>> Saved Reward Details Table to {self.reward_dir}")