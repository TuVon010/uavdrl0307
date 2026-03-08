import numpy as np
#rl-gym
class Base:
    def __init__(self):
        # =========================
        # 1. 场景设置 (3 UAV, 10 Users)
        # =========================
        self.n_uavs = 3
        self.n_users = 10
        self.field_X = [0, 1000]
        self.field_Y = [0, 1000]
        self.h = 50              # UAV高度 (m)
        self.time_step = 1.0      # delta_t (s)
        self.coverage_radius = 150.0

        # 初始化布局
        self.uav_init_radius = 350.0    # 无人机等边三角形外接圆半径 (距地图中心)
        self.user_cluster_radius = 50.0  # 用户高斯聚类标准差 (地图中心密集区)
        
        # =========================
        # 2. 归一化参数 (RL收敛核心)
        # =========================
        self.norm_pos = 1000.0
        self.norm_data = 5e6              
        self.norm_cycle = 1000.0          
        self.norm_freq = 10e9             
        
        self.norm_energy_uav = 600.0      
        self.norm_energy_user = 5.0       

        # =========================
        # 3. 通信与信道参数 (Urban LoS)
        # =========================
        self.f_c = 2e9            
        self.c = 3e8              
        self.alpha_los = 2.8      
        self.alpha_nlos = 3.5     
        self.a = 9.61
        self.b = 0.16
        self.beta0 = 10**(-60/10)
        self.sigma2 = 1e-13
        self.B_total = 20e6

        # =========================
        # 4. 智能体物理属性
        # =========================
        # --- UAV ---
        self.uav_v_max = 15.0
        self.C_uav = 10e9
        self.xi_m = 8.2e-10
        
        # 飞行能耗参数 (Rotary-Wing)
        self.P0 = 79.86
        self.Pi = 88.63
        self.U_tip = 120
        self.v0 = 4.03
        self.d0 = 0.6
        self.rho = 1.225
        self.s = 0.05
        self.A_rotor = 0.5

        # --- User (MU) ---
        self.p_tx_max = 0.2
        self.C_local = 1e9
        self.k_local = 1e-28
        
        # =========================
        # 5. Gauss-Markov 移动模型参数
        # =========================
        self.mobility_slot = 1.0
        self.user_mean_velocity = 0.5
        self.user_mean_direction = 0.1
        self.user_memory_level_velocity = 0.6  
        self.user_memory_level_direction = 0.8 
        self.user_Gauss_variance_velocity = 0.5
        self.user_Gauss_variance_direction = 0.5

        # =========================
        # 6. 任务生成
        # =========================
        self.task_size_min = 1e6  
        #self.task_size_max = 5e6
        self.task_size_max = 2e6
        self.cycles_min = 500
        self.cycles_max = 1000
        self.latency_max = 3.0

        # =========================
        # 7. 成本函数权重 (对应论文 eq.28)
        # =========================
        self.mu_L = 1.0
        self.mu_E = 0.5

        self.omega_H = 1.2
        self.omega_L = 1.0

        # =========================
        # 8. RL 奖励权重
        # =========================
        self.w_penalty = 2.0       # 超时惩罚权重
        self.w_overboundary = 0.5        # 无人机越界惩罚
        self.w_collision = 1.0     # 无人机碰撞惩罚

        self.w1_user = 0.4         # 用户奖励中系统奖励权重
        self.w2_user = 0.6         # 用户奖励中个体成本权重
        self.w_sys_uav = 0.3       # 无人机奖励中系统奖励权重
        self.w_ind_uav = 0.7       # 无人机奖励中个体奖励权重

        # self.w_proximity = 2.0     # 无人机接近用户的奖励权重
        # self.w_coverage = 1.5      # 无人机覆盖用户的奖励权重
        self.w_proximity = 0.5     # 无人机接近用户的奖励权重
        #self.w_coverage = 0.5      # 无人机覆盖用户的奖励权重
        self.w_coverage = 0      # 无人机覆盖用户的奖励权重

        # =========================
        # 9. 安全约束
        # =========================
        self.uav_safe_dist = 20.0
        self.boundary_warn = 50.0
