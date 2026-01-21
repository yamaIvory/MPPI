import torch
import numpy as np
from dynamics_gpu import DynamicsGPU

class MPPIControllerGPU:
    def __init__(self, urdf_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dyn = DynamicsGPU(urdf_path, device=self.device)
        
        # ---- Hyperparameters ----
        self.K = 400            # 400
        self.N = 10             # 20
        self.dt = 0.05          # 0.05
        self.lambda_ = 0.1      # 0.1
        self.alpha = 0.1        # 0.2

        # [Running Cost Weights] - 가는 과정
        self.w_pos = 100.0    # 100
        self.w_rot = 10.0     # 10
        self.w_vel = 0.1      # 0.1
        
        # [Terminal Cost Weights] - 최종 결과 (중요! 더 높게 설정)
        self.w_pos_term = 2000.0 
        self.w_rot_term = 200.0

        # Noise
        self.sigma = torch.tensor([0.01] * 6, device=self.device)  # 0.3
        
        self.U = torch.zeros((self.N, 6), device=self.device)

    def _compute_cost(self, ee_pos, ee_rot, P_goal, R_goal, u_t=None, is_terminal=False):
        """Cost 계산 함수 분리 (재사용을 위해)"""
        # 1. 위치 오차
        pos_err = torch.norm(ee_pos - P_goal, dim=1)**2
        
        # 2. 회전 오차
        R_diff = torch.matmul(R_goal.transpose(-1, -2), ee_rot)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        rot_err = 3.0 - trace
        
        # 가중치 선택
        w_p = self.w_pos_term if is_terminal else self.w_pos
        w_r = self.w_rot_term if is_terminal else self.w_rot
        
        cost = w_p * pos_err + w_r * rot_err
        
        if not is_terminal and u_t is not None:
            # 속도 비용 (에너지)
            cost += self.w_vel * torch.sum(u_t**2, dim=1)
            
            # 충돌 비용 (Running cost에서만 체크해도 충분)
            floor_mask = ee_pos[:, 2] < 0.05 
            cost = torch.where(floor_mask, cost + 10000.0, cost)
            
        return cost

    def get_optimal_command(self, q_curr_np, P_goal_np, R_goal_np):
        q_curr = torch.tensor(q_curr_np, device=self.device).float()
        q_sim = q_curr.unsqueeze(0).repeat(self.K, 1) 
        
        P_goal = torch.tensor(P_goal_np, device=self.device).float().unsqueeze(0)
        R_goal = torch.tensor(R_goal_np, device=self.device).float().unsqueeze(0)
        
        noise = torch.randn((self.K, self.N, 6), device=self.device) * self.sigma
        u_samples = self.U.unsqueeze(0) + noise
        #-------!!!!!!!!!!!!안전장치!!!!!!!!!!!!!!!!!!----------------------------
        limit_vel = 0.01   # 0.05
        u_samples = torch.clamp(u_samples, -limit_vel, limit_vel)
        #------------------------------------------------------------------------

        total_costs = torch.zeros(self.K, device=self.device)

        # 1. Running Cost (0 ~ N-1)
        for t in range(self.N):
            u_t = u_samples[:, t, :]
            q_next, ee_pos, ee_rot = self.dyn.step(q_sim, u_t)
            
            step_cost = self._compute_cost(ee_pos, ee_rot, P_goal, R_goal, u_t, is_terminal=False)
            total_costs += step_cost * self.dt
            
            q_sim = q_next # 상태 업데이트

        # 2. Terminal Cost (at N) - [여기가 부활했습니다!]
        # 마지막 q_sim은 N번째 스텝의 관절 각도입니다.
        # FK를 한 번 더 수행해서 마지막 위치를 확인합니다.
        tg = self.dyn.chain.forward_kinematics(q_sim)
        m = tg.get_matrix()
        term_pos = m[:, :3, 3]
        term_rot = m[:, :3, :3]
        
        term_cost = self._compute_cost(term_pos, term_rot, P_goal, R_goal, None, is_terminal=True)
        total_costs += term_cost # 마지막에 강력한 한 방 추가!

        # 3. Update
        beta = torch.min(total_costs)
        weights = torch.exp(-1.0/self.lambda_ * (total_costs - beta))
        weights /= (torch.sum(weights) + 1e-10)
        
        delta_u = torch.sum(weights.view(self.K, 1, 1) * noise, dim=0)
        self.U = (1 - self.alpha) * self.U + self.alpha * (self.U + delta_u)
        
        u_ret = self.U[0].clone()
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = 0.0
        
        return u_ret.cpu().numpy()
