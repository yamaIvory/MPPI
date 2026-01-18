import numpy as np
from dynamics import Dynamics

class MPPIController:
    def __init__(self, urdf_path):
        self.dyn = Dynamics(urdf_path)

        # ---- MPPI Hyperparameters ----
        self.K = 500            # number of rollouts
        self.N = 30             # horizon
        self.dt = 0.02
        self.lambda_ = 0.6      # temperature
        self.alpha = 0.3

        # Cost weights
        self.w_pos = 150.0
        self.w_rot = 20.0
        self.w_pos_terminal = 300.0
        self.w_rot_terminal = 50.0
        self.w_vel = 0.1

        # ---- Noise covariance ----
        self.sigma = np.array([1.0]*3 + [0.5]*3)
        self.sigma_sq = self.sigma**2

        # R = lambda * Sigma^{-1}
        self.R = np.diag(self.lambda_ / self.sigma_sq)

        # Nominal control sequence
        self.U = np.zeros((self.N, 6))

        # 관절 프레임 ID 찾기
        self.joint_names = ["BASE", "SHOULDER", "ARM", "FOREARM", "LOWER_WRIST", "UPPER_WRIST", "DUMMY"]
        self.joint_ids = []
        for name in self.joint_names:
            if self.dyn.model.existFrame(name):
                for i, frame in enumerate(self.dyn.model.frames):
                    if frame.name == name:
                        self.joint_ids.append(i)
                        break
        self.desk_height = 0.0                

    # ---------------------------------------------------
    # State cost q(x)
    # ---------------------------------------------------
    def state_cost(self, ee_pos, ee_rot, P_goal, R_goal):
        pos_err = np.linalg.norm(ee_pos - P_goal)
        rot_err = 3.0 - np.trace(R_goal.T @ ee_rot)
        
        cost = (self.w_pos * pos_err**2) + (self.w_rot * rot_err)
        return cost
   
    # ---------------------------------------------------
    # Terminal cost φ(x_T)
    # ---------------------------------------------------
    def terminal_cost(self, ee_pos, ee_rot, P_goal, R_goal):
        pos_err = np.linalg.norm(ee_pos - P_goal)
        rot_err = 3.0 - np.trace(R_goal.T @ ee_rot)

        return self.w_pos_terminal*pos_err**2 + self.w_rot_terminal*rot_err

    # --------------------------------------------------- 
    # Height Cost (바닥 충돌 방지)
    # ---------------------------------------------------
    def get_all_joint_height_cost(self, data):
        total_penalty = 0
        # 모든 주요 관절의 높이를 확인
        for frame_id in self.joint_ids:
            z_pos = data.oMf[frame_id].translation[2]
            
            # [핵심 로직]
            # 높이가 0.0보다 낮으면(지하) 엄청난 벌점
            if z_pos < self.desk_height:
                total_penalty += 1e9 
        return total_penalty
    # --------------------------------------------------- 
    # 관절 한계 넘어서는거 방지
    # ---------------------------------------------------
    def get_joint_limit_cost(self, q):
        """
        관절이 한계(Min/Max)에 가까워지거나 넘어가면 비용을 부과합니다.
        """
        # 안전 마진 (예: 한계치 5도 전부터 비용 발생 시작)
        margin = 0.05  # rad (약 3도)
        
        if np.any(q < self.dyn.q_min) or np.any(q > self.dyn.q_max):
            return 1e9  # 즉시 사망 (경로 폐기)
        
        # 1. 하한선(Min) 침범 검사: (q_min + margin) 보다 작으면 비용 발생
        # q가 q_min보다 작아질수록 값이 커짐 (0보다 클 때만 제곱)
        diff_lower = (self.dyn.q_min + margin) - q
        cost_lower = np.sum(np.maximum(0, diff_lower)**2)
        
        # 2. 상한선(Max) 침범 검사: (q_max - margin) 보다 크면 비용 발생
        diff_upper = q - (self.dyn.q_max - margin)
        cost_upper = np.sum(np.maximum(0, diff_upper)**2)

        # 가중치(w_limit)를 곱해서 반환 (이 값은 튜닝 필요, 보통 100~1000 등 크게 줌)
        w_limit = 100.0
        return w_limit * (cost_lower + cost_upper)
    # ---------------------------------------------------
    # MPPI main routine
    # ---------------------------------------------------
    def get_optimal_command(self, q_curr, P_goal, R_goal):

        # Noise: epsilon_k,t
        noise = np.random.normal(loc=0.0, scale=self.sigma,
                                size=(self.K, self.N, 6) )
        costs = np.zeros(self.K)

        # ---------------------------------------------------
        # Rollouts
        # ---------------------------------------------------
        for k in range(self.K):
            q_sim = q_curr.copy()
            S = 0.0

            for t in range(self.N):
                
                u_nom = self.U[t]
                du = noise[k, t]
                u = u_nom + du

                # ----------안전장치--------------------------
                v_limit = 0.5
                w_limit = 2.0
                u[:3] = np.clip(u[:3], -v_limit, v_limit)
                u[3:] = np.clip(u[3:], -w_limit, w_limit)
                #---------------------------------------------

                # Dynamics step
                q_next, ee_pos, ee_rot, _ = self.dyn.step(q_sim, u)

                # 책상 충돌 패널티
                height_cost = self.get_all_joint_height_cost(self.dyn.data)
                # 관절 한계 초과 패널티
                limit_cost = self.get_joint_limit_cost(q_next)
                # Cost 계산
                state_cost = self.state_cost(ee_pos, ee_rot, P_goal, R_goal)
                control_cost = self.w_vel*np.sum(u**2)

                # summation
                S += (state_cost + control_cost) * self.dt + height_cost + limit_cost

                if height_cost > 1e8:
                    S += 1e9 * (self.N - t) # 남은 시간만큼 벌점 추가
                    break
                if limit_cost > 1e8:
                    S += 1e9 * (self.N - t) # 남은 시간만큼 벌점 추가
                    break

                q_sim = q_next

            # Terminal cost
            if S < 1e8:   # 충돌 안 난 경우만
                S += self.terminal_cost(ee_pos, ee_rot, P_goal, R_goal)

            costs[k] = S

        # ---------------------------------------------------
        # Weight computation
        # ---------------------------------------------------
        beta = np.min(costs)

        weights = np.exp(-(costs - beta) / self.lambda_)
        weights /= np.sum(weights) + 1e-10

        # weight * noise
        delta_U = np.sum(weights[:, None, None] * noise, axis=0)
        
        # U_new = U_old + weight*noise
        U_new  = self.U + delta_U

        # smoothing
        self.U = (1-self.alpha)*self.U + self.alpha * U_new

        # ------------안전장치--------------------------------
        v_limit = 0.5
        w_limit = 2.0
        self.U[:, :3] = np.clip(self.U[:, :3], -v_limit, v_limit)
        self.U[:, 3:] = np.clip(self.U[:, 3:], -w_limit, w_limit)

        #----------------------------------------------------
        # Extract control & shift
        # ---------------------------------------------------
        u_opt = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = np.zeros(6)


        return u_opt