import os
import pinocchio as pin
import numpy as np

class Dynamics:
    def __init__(self, urdf_path):
      # 1. 로봇 모델 및 충돌 모델 로드
      self.model = pin.buildModelFromUrdf(urdf_path)
      self.data = self.model.createData()

      package_dir = os.path.dirname(os.path.abspath(urdf_path))
      self.collision_model = pin.buildGeomFromUrdf(
          self.model, urdf_path, pin.GeometryType.COLLISION, package_dirs=[package_dir]
      )
      
      # 2. 충돌 쌍 등록
      self.collision_model.addAllCollisionPairs()

      # 3. 인접 관절들을 충돌 쌍에서 삭제
      self.remove_adjacent_links_from_collision()
      
      # 4. 필터링이 끝난 '최종 모델'을 바탕으로 데이터 객체를 생성합니다.
      self.collision_data = pin.GeometryData(self.collision_model)

      # 나머지 설정
      self.ee_frame_id = self.model.getFrameId("DUMMY") 
      self.dt = 0.02
      self.damping = 1e-4
      self.q_min = self.model.lowerPositionLimit
      self.q_max = self.model.upperPositionLimit


    def remove_adjacent_links_from_collision(self):
        """로봇의 트리 구조를 분석하여 인접한 링크 쌍을 충돌 검사에서 제외합"""
        to_remove = []
        for i, pair in enumerate(self.collision_model.collisionPairs):
            obj1 = self.collision_model.geometryObjects[pair.first]
            obj2 = self.collision_model.geometryObjects[pair.second]
            
            joint1 = obj1.parentJoint
            joint2 = obj2.parentJoint
            
            if joint1 == joint2 or \
               self.model.parents[joint1] == joint2 or \
               self.model.parents[joint2] == joint1:
                to_remove.append(i)
            # 같은 관절에 있거나 부모자식 관계라면 충돌 쌍에서 제외
        for i in reversed(to_remove):
            self.collision_model.removeCollisionPair(self.collision_model.collisionPairs[i])
            
        print(f"계층 구조 분석을 통해 {len(to_remove)}개의 인접 링크들을 충돌 쌍에서 제외")


    def check_self_collision(self, q):
        """현재 각도 q에서 자체 충돌 여부를 반환합니다."""
        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, q)
        # stop_at_first=True 로 설정하여 첫 충돌 발견 시 바로 반환
        return pin.computeCollisions(self.collision_model, self.collision_data, True)


    def solve_ik(self, q, u_task):
        """
        현재 각도 q와 목표 속도 u_task를 받아,
        물리적 한계를 고려한 안전한 관절 속도 dq를 반환합니다.
        """
        # 1. 자코비안 계산을 위한 업데이트
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # 2. 자코비안 가져오기 (작업공간 6자유도 x 관절 n개)
        J_full = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:, :6] 
        
        # 3. Damped Least Squares (DLS) IK 풀이
        JJT = J @ J.T
        damp_matrix = (self.damping ** 2) * np.eye(6)
        temp = np.linalg.solve(JJT + damp_matrix, u_task)
        dq_arm = J.T @ temp

        # 4. [Hardware Safety] 관절 속도 물리적 한계 클리핑
        # Kinova Gen3 Lite 스펙상 한계 혹은 안전 한계 설정
        #-----------안전장치-----------------------------------------
        joint_vel_limit = 0.2  # rad/s
        dq_arm = np.clip(dq_arm, -joint_vel_limit, joint_vel_limit)
        #-----------------------------------------------------------

        # 5. 전체 dq 벡터 생성 (그리퍼 포함)
        dq = np.zeros(self.model.nq)
        dq[:6] = dq_arm 
        
        return dq

    def step(self, q_curr, u_task):
        """제어 입력을 받아 다음 관절 상태를 계산합니다."""
        dq = self.solve_ik(q_curr, u_task)
        q_next = pin.integrate(self.model, q_curr, dq * self.dt)

        #-----------안전장치------------------------------------
        q_next = np.clip(q_next, self.q_min, self.q_max)
        #------------------------------------------------------
        
        pin.forwardKinematics(self.model, self.data, q_next)
        pin.updateFramePlacements(self.model, self.data)
        frame = self.data.oMf[self.ee_frame_id]
        
        return q_next, frame.translation, frame.rotation, dq