import numpy as np
import time
import os
import pinocchio as pin
import meshcat.geometry as g
import meshcat.transformations as tf

# Pinocchio의 시각화 도구
from pinocchio.visualize import MeshcatVisualizer
from mppi_solver import MPPIController

def run_simulation():
    # ---------------------------------------------------------
    # 1. 경로 및 설정
    # ---------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "gen3_lite.urdf")
    mesh_dir = current_dir # 메쉬 파일이 있는 기준 폴더

    print(f"1. 시뮬레이션 초기화 중...")
    
    # ---------------------------------------------------------
    # 2. 로봇 & 시각화 도구 로딩
    # ---------------------------------------------------------
    # MPPI 컨트롤러 생성
    mppi = MPPIController(urdf_path)
    model = mppi.dyn.model
    
    # 시각화용 모델 생성 (Mesh 파일 경로 지정)
    visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL, package_dirs=mesh_dir)
    collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION, package_dirs=mesh_dir)

    # Meshcat 뷰어 연결
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    
    try:
        viz.initViewer(open=True) # 자동으로 브라우저 열기
    except ImportError:
        print("Error: Meshcat을 열 수 없습니다.")
        return

    viz.loadViewerModel()
    
    print("-> 3D 뷰어 로딩 완료! (브라우저를 확인하세요)")

    # ---------------------------------------------------------
    # 3. 목표물(빨간 공) 시각화 추가
    # ---------------------------------------------------------
    # 목표 위치 설정 (시작 위치에서 Z축 +20cm)
    q_curr = np.array([0,0,0,0,0,0]) 
    _, start_P, _, _ = mppi.dyn.step(q_curr, np.zeros(6))
    
    target_P = np.array([0.3, 0.3, 0.3])
    target_R = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    # 빨간 공 생성 (반지름 2cm)
    viz.viewer['target_ball'].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=0xff0000, opacity=0.8))
    viz.viewer['target_ball'].set_transform(tf.translation_matrix(target_P))

    # 손끝(EE)을 표시할 파란 공 생성 (반지름 1.5cm)
    viz.viewer['ee_ball'].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x0000ff, opacity=0.8))
    viz.viewer['target_frame'].set_object(g.triad(0.1)) # 목표 지점 화살표 (크기 0.1m)
    viz.viewer['ee_frame'].set_object(g.triad(0.1))     # 내 손끝 화살표

    print(f"\n=== 시뮬레이션 시작 ===")
    print(f"목표 위치(빨간 공): {target_P}")

    # ---------------------------------------------------------
    # 4. 루프 실행
    # ---------------------------------------------------------
    dt = 0.02
    max_steps = 200 # 4초 정도 실행
    
    # 초기 자세 보여주기
    viz.display(q_curr)
    time.sleep(1.0) # 1초 대기

    try:
        for step in range(max_steps):
            loop_start = time.time()
            
            # (1) MPPI 계산
            u_opt = mppi.get_optimal_command(q_curr, target_P, target_R)
            
            # (2) 로봇 이동
            q_next, curr_P, _, _ = mppi.dyn.step(q_curr, u_opt)
            q_curr = q_next
            
            # (3) 화면 업데이트 (가장 중요!)
            viz.display(q_curr)
            viz.viewer['ee_ball'].set_transform(tf.translation_matrix(curr_P))
            T_target = np.eye(4)
            T_target[:3, 3] = target_P
            T_target[:3, :3] = target_R
            viz.viewer['target_frame'].set_transform(T_target)

            # 내 손끝 화살표 위치/자세 업데이트
            pin.framesForwardKinematics(model, mppi.dyn.data, q_curr)
            ee_id = model.getFrameId("DUMMY")
            curr_T = mppi.dyn.data.oMf[ee_id] # 현재 변환 행렬 (4x4)
            
            # Meshcat은 numpy 배열을 원하므로 .np 변환
            viz.viewer['ee_frame'].set_transform(curr_T.np)


            # 회전 확인
            # 현재 로봇 손끝의 회전 행렬 가져오기 (아까 화살표 그릴 때 쓴 curr_T에서 추출)
            curr_R = curr_T.rotation 
            
            # Trace Trick: 두 회전이 같으면 0에 가까워짐
            # (완벽 일치하면 0, 90도 틀어지면 1, 180도 반대면 4)
            R_diff = target_R.T @ curr_R
            rot_err = 3.0 - np.trace(R_diff)

            # 거리 확인
            dist = np.linalg.norm(curr_P - target_P)
            if step % 10 == 0:
                print(f"[Step {step}] 목표까지 거리: {dist:.4f}m 각도 차이: {rot_err:.4f}")

            if dist < 0.02 and rot_err < 0.1:
                print(f"\n목표 도달! step: {step} 소요시간: {step*dt:.4f}")
                break

            # 속도 조절
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n종료합니다.")

if __name__ == "__main__":
    run_simulation()