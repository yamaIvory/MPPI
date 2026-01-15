#!/usr/bin/env python3

import sys
import rospy
import numpy as np
import os
import time
from std_msgs.msg import Float64MultiArray
from kortex_driver.srv import *
from kortex_driver.msg import *

# 사용자 정의 MPPI Solver
try:
    from mppi_solver import MPPIController
except ImportError:
    rospy.logerr("mppi_solver.py를 찾을 수 없습니다. 파일 경로를 확인하세요.")
    sys.exit()

class Gen3LiteMPPINode:
    def __init__(self):
        try:
            rospy.init_node('gen3_lite_mppi_integrated_node')

            # 1. 파라미터 및 설정
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            self.HOME_ACTION_IDENTIFIER = 2
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.urdf_path = os.path.join(current_dir, "gen3_lite.urdf")
            
            # 상태 변수
            self.last_action_notif_type = None
            self.q_curr = None
            self.is_init_success = False

            # 2. MPPI 컨트롤러 초기화
            if not os.path.exists(self.urdf_path):
                rospy.logerr(f"URDF 파일을 찾을 수 없습니다: {self.urdf_path}")
                sys.exit()
            self.mppi = MPPIController(self.urdf_path)

            # 3. Kortex 서비스 프록시 설정
            rospy.loginfo(f"[{self.robot_name}] 서비스 연결 중...")
            self.setup_services()

            # 4. ROS 통신 설정 (구독 및 발행)
            # 액션 상태 알림 구독
            self.action_topic_sub = rospy.Subscriber(
                f"/{self.robot_name}/action_topic", ActionNotification, self.cb_action_topic)
            
            # 실시간 로봇 피드백 구독 (Degree로 들어옴)
            self.sub_feedback = rospy.Subscriber(
                f"/{self.robot_name}/base_feedback", BaseCyclic_Feedback, self.cb_joint_feedback)
            
            # 실시간 관절 속도 명령 발행 (Degree/s 단위 권장)
            self.pub_vel = rospy.Publisher(
                f"/{self.robot_name}/joint_group_velocity_controller/command", 
                Float64MultiArray, queue_size=1)

            # 종료 시 안전 장치
            rospy.on_shutdown(self.stop_robot)
            self.is_init_success = True
            rospy.loginfo("✅ 모든 시스템 초기화 완료")

        except Exception as e:
            rospy.logerr(f"초기화 중 오류 발생: {e}")
            self.is_init_success = False

    def setup_services(self):
        """필수 서비스 서버 대기 및 프록시 생성"""
        prefix = f"/{self.robot_name}"
        services = {
            'clear_faults': (prefix + '/base/clear_faults', Base_ClearFaults),
            'read_action': (prefix + '/base/read_action', ReadAction),
            'execute_action': (prefix + '/base/execute_action', ExecuteAction),
            'set_ref_frame': (prefix + '/control_config/set_cartesian_reference_frame', SetCartesianReferenceFrame),
            'activate_notif': (prefix + '/base/activate_publishing_of_action_topic', OnNotificationActionTopic)
        }
        for name, (path, srv_type) in services.items():
            rospy.wait_for_service(path, timeout=5.0)
            setattr(self, name, rospy.ServiceProxy(path, srv_type))

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def cb_joint_feedback(self, msg):
        """로봇의 관절 각도를 수신하여 라디안으로 변환"""
        q_deg = [msg.actuators[i].position for i in range(6)]
        self.q_curr = np.deg2rad(q_deg)

    def wait_for_action_end(self, timeout=15.0):
        """액션 완료 대기 루프"""
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.last_action_notif_type == ActionEvent.ACTION_END:
                return True
            if self.last_action_notif_type == ActionEvent.ACTION_ABORT:
                return False
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                return False
            rospy.sleep(0.01)

    def stop_robot(self):
        """종료 시 모든 관절 속도를 0으로 설정하여 로봇 정지"""
        rospy.logwarn("⚠️ 노드 종료: 로봇 정지 명령 전송")
        msg = Float64MultiArray(data=[0.0] * 6)
        self.pub_vel.publish(msg)

    def prepare_hardware(self):
        """MPPI 전 하드웨어 안전 점검 및 홈 이동"""
        rospy.loginfo("1. 결함(Faults) 제거 중...")
        self.clear_faults()
        rospy.sleep(2.0)

        rospy.loginfo("2. Home 위치로 이동 시작...")
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        res = self.read_action(req)
        
        exec_req = ExecuteActionRequest()
        exec_req.input = res.output
        self.last_action_notif_type = None
        self.execute_action(exec_req)
        
        if not self.wait_for_action_end():
            rospy.logerr("Home 이동 실패")
            return False

        rospy.loginfo("3. 좌표계 설정 (Base Frame)...")
        frame_req = SetCartesianReferenceFrameRequest()
        frame_req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE
        self.set_ref_frame(frame_req)
        
        # 알림 활성화
        self.activate_notif(OnNotificationActionTopicRequest())
        rospy.sleep(1.0)
        return True

    def run_mppi_loop(self, target_P, target_R):
        rospy.loginfo("🚀 MPPI 실시간 제어 루프 진입")
        rate = rospy.Rate(50) 
        
        # 속도 급변 방지를 위한 이전 속도 저장
        prev_dq = np.zeros(6)
        alpha = 0.7 # 필터 계수 (0~1, 높을수록 부드러움)

        while not rospy.is_shutdown():
            if self.q_curr is None: continue

            # 1. MPPI 계산
            u_opt = self.mppi.get_optimal_command(self.q_curr, target_P, target_R)
            
            # 2. dq 계산
            _, _, _, dq_rad = self.mppi.dyn.step(self.q_curr, u_opt)
            dq_deg = np.rad2deg(dq_rad)

            # 3. [보완] 속도 필터링 및 클램핑
            # 갑작스러운 튀는 명령 방지
            dq_deg = alpha * prev_dq + (1 - alpha) * dq_deg
            dq_deg = np.clip(dq_deg, -30.0, 30.0)
            prev_dq = dq_deg

            # 4. 도착 판정 (1cm)
            _, curr_P, _, _ = self.mppi.dyn.step(self.q_curr, np.zeros(6))
            dist = np.linalg.norm(curr_P - target_P)
            
            if dist < 0.01:
                dq_deg = np.zeros(6)
                rospy.loginfo_throttle(5, f"목표 도착 (오차: {dist:.4f}m)")

            # 5. 명령 발행
            msg = Float64MultiArray(data=dq_deg.tolist())
            self.pub_vel.publish(msg)
            
            rate.sleep()

    def main(self):
        if not self.is_init_success: return

        # 1단계: 하드웨어 준비 (Home 이동)
        if self.prepare_hardware():
            rospy.loginfo("✅ 하드웨어 준비 완료. 목표 설정 중...")
            
            # 2단계: 목표 지점 설정 (현재 위치에서 위로 10cm)
            while self.q_curr is None: rospy.sleep(0.1)
            _, start_P, start_R, _ = self.mppi.dyn.step(self.q_curr, np.zeros(6))
            
            target_P = start_P.copy()
            target_P[2] += 0.10 # 10cm 위
            target_R = start_R.copy() # 회전은 유지
            
            # 3단계: MPPI 실시간 루프 시작
            self.run_mppi_loop(target_P, target_R)

if __name__ == "__main__":
    node = Gen3LiteMPPINode()
    node.main()