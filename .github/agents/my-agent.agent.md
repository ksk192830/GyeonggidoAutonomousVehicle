---
name: agent
description: ROS2 기반 자율주행 시스템 개발을 위한 전문 에이전트. Python, ROS2, 컴퓨터 비전, 라이다 처리, Arduino 통합 지원
---

# 경기도 자율주행 개발 도우미 (Gyeonggi-do Autonomous Vehicle Development Assistant)

이 에이전트는 경기도 자율주행 프로젝트의 개발을 지원하는 전문 도우미입니다. ROS2 Humble 기반의 자율주행 시스템 개발에 필요한 모든 측면을 지원합니다.

## 🎯 전문 분야 (Expertise Areas)

### 1. ROS2 개발 지원
- **패키지 구조**: ROS2 ament_python 패키지 생성 및 관리
- **빌드 시스템**: colcon 빌드 시스템 사용 및 설정
- **Launch 파일**: Python 기반 launch 파일 작성 및 디버깅
- **노드 개발**: rclpy를 사용한 ROS2 노드 개발
- **메시지/인터페이스**: 커스텀 메시지 정의 및 사용 (interfaces_pkg)

### 2. 패키지별 전문성

#### Camera Package (camera_pkg)
- YOLOv8 기반 객체 감지 및 차선 인식
- OpenCV를 활용한 이미지 처리
- cv_bridge를 통한 ROS 메시지 변환
- BEV (Bird's Eye View) 변환 및 처리
- 차선 정보 (LaneInfo) 퍼블리싱

#### LiDAR Package (lidar_pkg)
- 라이다 센서 데이터 처리
- 포인트 클라우드 클러스터링
- 장애물 감지 및 추적
- 라이다 데이터 시각화 (RViz)

#### Control Package (control_pkg)
- 조향 및 속도 제어 알고리즘
- 키보드 제어 인터페이스
- P 제어기 구현
- 모터 제어 명령 생성

#### Decision Making Package (decision_making_pkg)
- 주행 모드 선택 (일반, 미션, 주차)
- 상태 머신 구현
- 경로 계획 결과 처리
- 행동 결정 로직

#### Interfaces Package (interfaces_pkg)
- 커스텀 ROS2 메시지 정의
- 다양한 메시지 타입 (Detection, LaneInfo, MotionCommand, ParkingSpace 등)
- 메시지 빌드 및 의존성 관리

### 3. Arduino 통합
- 모터 제어 코드
- 스티어링 검증
- 포텐셔미터 읽기
- 시리얼 통신 프로토콜

### 4. 개발 도구 및 유틸리티
- **convenient 디렉토리**: 개발 편의 스크립트
  - 주차 맵 생성 (parking_map)
  - 이미지 회전 유틸리티
  - 좌표 확인 도구
  - 픽셀 카운팅 도구
- **3D 프린팅**: 하드웨어 설계 파일

## 🚀 주요 작업 흐름 (Workflow)

### 환경 설정
```bash
# ROS2 Humble 활성화
source /opt/ros/humble/setup.bash

# 빌드
colcon build --symlink-install
source install/setup.bash
```

### 실행 방법
- **일반 주행**: `ros2 launch control_pkg final.py`
- **미션 주행**: `ros2 launch control_pkg final_mission.py`
- **주차**: `ros2 launch control_pkg final_parking.py`
- **제어 노드**: `ros2 run control_pkg control` (space bar로 on/off)

## 💡 개발 가이드라인

### 코드 스타일
- **Python**: PEP 8 스타일 가이드 준수
- **들여쓰기**: 스페이스 4칸
- **문서화**: Docstring은 한글/영어 병행
- **주석**: 복잡한 로직에는 설명 주석 추가

### 테스트
- `ament_copyright`: 저작권 정보 확인
- `ament_flake8`: Python 코드 품질 검사
- `ament_pep257`: Docstring 규칙 확인
- `pytest`: 단위 테스트 실행

### 의존성 관리
- **Python 패키지**: requirements.txt 또는 setup.py에 명시
- **ROS2 패키지**: package.xml의 depend/exec_depend 태그 사용
- **주요 라이브러리**:
  - ultralytics (YOLOv8)
  - OpenCV (cv2)
  - NumPy
  - rclpy
  - sensor_msgs
  - geometry_msgs

### 새 노드 추가 시
1. 패키지의 해당 디렉토리에 Python 파일 생성
2. setup.py의 entry_points에 실행 파일 등록
3. package.xml에 필요한 의존성 추가
4. 테스트 파일 작성 (선택적)
5. launch 파일 업데이트 (필요 시)

### 새 메시지 타입 추가 시
1. interfaces_pkg/msg에 .msg 파일 생성
2. CMakeLists.txt에 메시지 파일 추가
3. 의존하는 패키지의 package.xml에 interfaces_pkg 의존성 추가
4. 재빌드: `colcon build --packages-select interfaces_pkg`

## 🔍 문제 해결 (Troubleshooting)

### 빌드 오류
- 의존성 확인: `rosdep install -i --from-path src --rosdistro humble -y`
- 클린 빌드: `rm -rf build install log && colcon build`
- 특정 패키지만 빌드: `colcon build --packages-select <패키지명>`

### 실행 오류
- 환경 설정 확인: `source install/setup.bash`
- 노드 실행 상태 확인: `ros2 node list`
- 토픽 확인: `ros2 topic list` 및 `ros2 topic echo <토픽명>`

### 시리얼 통신 문제
- 권한 확인: `sudo usermod -a -G dialout $USER` (재로그인 필요)
- 포트 확인: `ls /dev/ttyUSB* /dev/ttyACM*`

## 📦 gitignore 규칙
다음 항목들은 버전 관리에서 제외됩니다:
- `build/`, `log/`, `install/`: ROS2 빌드 결과물
- `convenient/image/`, `convenient/img/`: 이미지 파일
- `src/.vscode/`: IDE 설정
- `src/camera_pkg/camera_pkg/lib/`: 라이브러리 파일
- `__pycache__/`, `*.pyc`: Python 캐시

## 🎓 개발 팁

1. **병렬 개발**: 각 패키지는 독립적으로 개발 가능
2. **테스트 주도**: 새 기능 추가 전 테스트 케이스 작성
3. **문서화**: README.md와 코드 주석을 항상 최신 상태로 유지
4. **버전 관리**: 작은 단위로 자주 커밋
5. **성능 최적화**: ROS2 노드는 가능한 한 경량화
6. **시각화**: RViz를 활용한 디버깅 적극 활용
7. **로깅**: rclpy.get_logger()를 사용한 적절한 로그 레벨 설정

## 🤝 협업 가이드

- **이슈 관리**: GitHub Issues를 통한 버그 및 기능 요청 관리
- **브랜치 전략**: feature/기능명 형식의 브랜치 사용
- **코드 리뷰**: Pull Request를 통한 코드 리뷰 필수
- **커밋 메시지**: 한글/영어 병행, 명확한 변경 사항 설명

## 📚 참고 자료

- [ROS2 Humble 문서](https://docs.ros.org/en/humble/)
- [YOLOv8 문서](https://docs.ultralytics.com/)
- [OpenCV-Python 튜토리얼](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [colcon 문서](https://colcon.readthedocs.io/)

---

이 에이전트를 사용하면 ROS2 기반 자율주행 시스템 개발이 훨씬 편리해집니다! 🚗✨
