import numpy as np
from dynamics import Dynamics


class MPPIController:
    def __init__(self, urdf_path):
        self.dyn = Dynamics(urdf_path)

        # --- MPPI Hyperparameters ---
        self.K = 500            # number of rollouts
        self.N = 40             # horizon
        self.dt = 0.05
        self.lambda_ = 0.05      # temperature
        self.nu = 1000.0         # importance sampling coefficient

        # Cost weights
        self.w_pos = 100000.0
        self.w_rot = 500.0
        self.w_pos_terminal = 500000.0
        self.w_rot_terminal = 10000.0

        self.alpha = 0.5

        # --- Noise covariance ---
        self.sigma = np.array([0.4]*3 + [0.2]*3)
        self.sigma_sq = self.sigma**2

        # R = lambda * Sigma^{-1}   (B = I 가정)
        self.R = np.diag(self.lambda_ / self.sigma_sq)

        # Nominal control sequence
        self.U = np.zeros((self.N, 6))

        # 관절 이름 부여 ( 책상 충돌 방지 확인을 위해 )
        self.joint_names = [
            "BASE", "SHOULDER", "ARM", "FOREARM", 
            "LOWER_WRIST", "UPPER_WRIST", "DUMMY"
        ]

        self.joint_ids = []
        for name in self.joint_names:
            if self.dyn.model.existFrame(name):
                # [해결책] getFrameId는 중복 시 에러를 던질 수 있으므로 
                # 직접 프레임 목록을 돌며 첫 번째로 일치하는 ID만 가져옵니다.
                for i, frame in enumerate(self.dyn.model.frames):
                    if frame.name == name:
                        self.joint_ids.append(i)
                        break 
            else:
                print(f"⚠️ 경고: '{name}' 프레임을 찾을 수 없습니다.")

        self.desk_height = 0.05  # 안전 마진 5cm 설정
    # ---------------------------------------------------
    # State cost q(x)
    # ---------------------------------------------------
    def state_cost(self, ee_pos, ee_rot, P_goal, R_goal):
        # 위치 비용
        pos_err = np.linalg.norm(ee_pos - P_goal)
        cost = self.w_pos * pos_err**2

        # 자세 비용
        # 완벽히 일치하면 Trace 값은 3, 가장 멀면 -1
        rot_err_mat = R_goal.T @ ee_rot
        trace_val = np.trace(rot_err_mat)
        
        # 0(일치) ~ 4(반대) 사이의 값을 가짐
        cost += self.w_rot * (3.0 - trace_val)

        return cost
    
    # ---------------------------------------------------
    # Control Cost   0.5*(1 - nu^-1)*du^T*R*d  + u^T*R*du + 0.5*u^T*R*u
    # ---------------------------------------------------
    def get_control_cost(self, u_nom, du):

        inv_nu = 1.0 / self.nu
        
        # du, u_nom은 모두 (6,) 크기여야 함
        
        term1 = 0.5 * (1.0 - inv_nu) * du.T @ self.R @ du
        term2 = u_nom.T @ self.R @ du
        term3 = 0.5 * u_nom.T @ self.R @ u_nom
        
        return term1
    # ---------------------------------------------------
    # Terminal cost φ(x_T)
    # ---------------------------------------------------
    def terminal_cost(self, ee_pos, ee_rot, P_goal, R_goal):

        pos_err = np.linalg.norm(ee_pos - P_goal)
        rot_err = 3.0 - np.trace(R_goal.T @ ee_rot)

        return self.w_pos_terminal*pos_err**2 + self.w_rot_terminal*rot_err
    
    # ---------------------------------------------------
    # Height Cost (책상 충돌)
    # ---------------------------------------------------
    def get_all_joint_height_cost(self, data):
        total_penalty = 0
        for frame_id in self.joint_ids:
            # 각 관절/프레임의 현재 전역 위치(Z) 가져오기
            z_pos = data.oMf[frame_id].translation[2]
            
            # 책상 높이보다 낮아지면 벌점 부여
            if z_pos < self.desk_height:
                # 낮아질수록 벌점이 제곱으로 증가 (강력한 반발력)
                total_penalty += (self.desk_height - z_pos)**2 * 1000000.0
                
        return total_penalty
    # ---------------------------------------------------
    # MPPI main routine
    # ---------------------------------------------------
    def get_optimal_command(self, q_curr, P_goal, R_goal):

        # Noise: epsilon_k,t
        noise = np.random.normal(
            loc=0.0,
            scale=self.sigma,
            size=(self.K, self.N, 6)
        )

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

                q_next, ee_pos, ee_rot, _ = self.dyn.step(q_sim, u)

                # 자체충돌
                if self.dyn.check_self_collision(q_next):
                    S += 1e12 * (self.N - t) # 즉시 무한대급 벌점 부여
                    break

                state_cost = self.state_cost(ee_pos, ee_rot, P_goal, R_goal)
                control_cost = self.get_control_cost(u_nom, du)
                collision_cost = self.get_all_joint_height_cost(self.dyn.data)

                # summation
                S += (state_cost + control_cost + collision_cost)*self.dt

                q_sim = q_next

            # Terminal cost
            S += self.terminal_cost(ee_pos, ee_rot, P_goal, R_goal)

            costs[k] = S

        # ---------------------------------------------------
        # Weight computation (with control quadratic term)
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

        # ---------------------------------------------------
        # Extract control & shift
        # ---------------------------------------------------
        u_opt = self.U[0].copy()

        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = np.zeros(6)

        return u_opt
