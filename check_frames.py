import pinocchio as pin
import sys

# URDF 파일 이름이 맞는지 확인하세요
urdf_filename = "gen3_lite.urdf"

try:
    model = pin.buildModelFromUrdf(urdf_filename)
    print(f"\n=== '{urdf_filename}'에 정의된 프레임 목록 ===")
    for i, frame in enumerate(model.frames):
        print(f"[{i}] {frame.name}")
    print("============================================\n")
    print("위 목록에서 'EndEffector'나 'Tool', 'Gripper' 같은 단어가 들어간")
    print("맨 마지막 쪽 이름을 찾으세요!")

except Exception as e:
    print(f"에러 발생: {e}")