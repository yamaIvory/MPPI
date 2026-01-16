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

                # Dynamics step
                q_next, ee_pos, ee_rot, _ = self.dyn.step(q_sim, u)

                # 책상 충돌
                height_cost = self.get_all_joint_height_cost(self.dyn.data)
                # Cost 계산
                state_cost = self.state_cost(ee_pos, ee_rot, P_goal, R_goal)

                # summation
                S += (state_cost + height_cost) * self.dt

                if height_cost > 1e8:
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

        # ---------------------------------------------------
        # Extract control & shift
        # ---------------------------------------------------
        u_opt = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = np.zeros(6)

        return u_opt