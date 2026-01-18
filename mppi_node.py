#!/usr/bin/env python3

import sys
import rospy
import numpy as np
import os
from std_msgs.msg import Float64MultiArray
from kortex_driver.srv import *
from kortex_driver.msg import *

# 사용자 정의 MPPI Solver
try:
    from mppi_solver import MPPIController
except ImportError:
    rospy.logerr("mppi_solver.py를 찾을 수 없습니다.")
    sys.exit()

class Gen3LiteMPPINode:
    def __init__(self):
        try:
            rospy.init_node('gen3_lite_mppi_integrated_node')

            # 1. 설정
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.urdf_path = os.path.join(current_dir, "gen3_lite.urdf")
            
            # 2. MPPI 컨트롤러 초기화
            self.mppi = MPPIController(self.urdf_path)
            self.nq = self.mppi.dyn.model.nq  # 이제 10입니다. 

            # 상태 변수 (10차원으로 초기화)
            self.q_curr = None
            self.is_init_success = False

            # 3. 서비스 및 통신 설정
            self.setup_services()
            self.action_topic_sub = rospy.Subscriber(f"/{self.robot_name}/action_topic", ActionNotification, self.cb_action_topic)
            self.sub_feedback = rospy.Subscriber(f"/{self.robot_name}/base_feedback", BaseCyclic_Feedback, self.cb_joint_feedback)
            self.pub_vel = rospy.Publisher(f"/{self.robot_name}/joint_group_velocity_controller/command", Float64MultiArray, queue_size=1)

            rospy.on_shutdown(self.stop_robot)
            self.is_init_success = True
            rospy.loginfo("✅ 시스템 초기화 완료 (Home 이동 생략)")

        except Exception as e:
            rospy.logerr(f"초기화 중 오류 발생: {e}")

    def setup_services(self):
        prefix = f"/{self.robot_name}"
        # Home 이동 관련 서비스는 리스트에서 제외하거나 호출하지 않습니다.
        services = {
            'clear_faults': (prefix + '/base/clear_faults', Base_ClearFaults),
            'set_ref_frame': (prefix + '/control_config/set_cartesian_reference_frame', SetCartesianReferenceFrame),
            'activate_notif': (prefix + '/base/activate_publishing_of_action_topic', OnNotificationActionTopic)
        }
        for name, (path, srv_type) in services.items():
            rospy.wait_for_service(path, timeout=5.0)
            setattr(self, name, rospy.ServiceProxy(path, srv_type))

    def cb_joint_feedback(self, msg):
        """로봇의 관절 각도를 수신 (6개 피드백 + 4개 가상 그리퍼)"""
        # 실제 로봇 팔 6축 
        q_arm = [msg.actuators[i].position for i in range(6)]
        
        # 10차원 상태 벡터 구성 
        q_full = np.zeros(self.nq)
        q_full[:6] = np.deg2rad(q_arm)
        # 그리퍼 4축은 0(고정)으로 채움
        q_full[6:] = 0.0
        
        self.q_curr = q_full

    def stop_robot(self):
        rospy.logwarn("⚠️ 로봇 정지")
        msg = Float64MultiArray(data=[0.0] * 6) # 명령은 항상 팔 6축만 보냄
        self.pub_vel.publish(msg)

    def prepare_hardware(self):
        rospy.loginfo("1. 결함(Faults) 제거...")
        self.clear_faults()
        
        rospy.loginfo("2. 좌표계 및 알림 설정...")
        frame_req = SetCartesianReferenceFrameRequest()
        frame_req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE
        self.set_ref_frame(frame_req)
        self.activate_notif(OnNotificationActionTopicRequest())
        
        rospy.sleep(1.0)
        return True

    def run_mppi_loop(self, target_P, target_R):
        rospy.loginfo("🚀 현재 위치에서 Z축 상승 시작")
        rate = rospy.Rate(50) 
        prev_dq = np.zeros(6)
        alpha = 0.6 

        while not rospy.is_shutdown():
            if self.q_curr is None: continue

            # 1. MPPI 계산
            u_opt = self.mppi.get_optimal_command(self.q_curr, target_P, target_R)
            
            # 2. dq 계산 (10차원 중 앞 6개만 사용) [cite: 1-13, 14-22]
            dq_rad_full = self.mppi.dyn.solve_ik(self.q_curr, u_opt)
            dq_arm = dq_rad_full[:6]

            # 3. 속도 필터링 및 안전 클램핑
            dq_arm = alpha * prev_dq + (1 - alpha) * dq_arm
            dq_arm = np.clip(dq_arm, -0.5, 0.5) # 안전을 위해 속도 제한 강화
            prev_dq = dq_arm

            # 4. 도착 판정
            _, curr_P, curr_R, _ = self.mppi.dyn.step(self.q_curr, np.zeros(6))
            dist = np.linalg.norm(curr_P - target_P)
            rot_err = 3.0 - np.trace(target_R.T @ curr_R)
            
            if dist < 0.02 and rot_err < 0.1:
                dq_arm = np.zeros(6)
                rospy.loginfo_throttle(10, "목표 높이 도달")

            # 5. 명령 발행 (팔 관절 6개)
            msg = Float64MultiArray(data=dq_arm.tolist())
            self.pub_vel.publish(msg)
            
            rate.sleep()

    def main(self):
        if not self.is_init_success: return

        if self.prepare_hardware():
            # 피드백 대기
            while self.q_curr is None and not rospy.is_shutdown():
                rospy.sleep(0.1)
            
            # 현위치 기반 목표 설정
            # DUMMY(손끝) 프레임을 기준으로 현재 위치를 계산합니다. [cite: 12-13]
            _, start_P, start_R, _ = self.mppi.dyn.step(self.q_curr, np.zeros(6))
            
            target_P = start_P.copy()
            target_P[2] += 0.10  # 현재 손끝 위치에서 10cm 위로 
            target_R = start_R.copy() 
            
            rospy.loginfo(f"📍 현재 높이: {start_P[2]:.3f}m -> 목표 높이: {target_P[2]:.3f}m")
            self.run_mppi_loop(target_P, target_R)

if __name__ == "__main__":
    node = Gen3LiteMPPINode()
    node.main()