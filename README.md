#  경기도 자율주행  🚗

ROS2 기반 자율주행 시스템

## 📦 Manual

---

## 0. 환경 설정

### ROS2 Humble 활성화

```bash
source /opt/ros/humble/setup.bash
```

### 빌드

```bash
colcon build --symlink-install
source install/setup.bash
```

---

## 1. 일반 주행 실행

```bash
ros2 launch control_pkg final.py
```


## 2. 미션 주행 실행

```bash
ros2 launch control_pkg final_mission.py
```


## 3. 주차 실행

```bash
ros2 launch control_pkg final_parking.py
```

---

## 🔧 추가 제어 노드 (space bar로 제어 on/off)

```bash
ros2 run control_pkg control
```


