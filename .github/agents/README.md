# Custom Agent 사용 가이드 (Custom Agent Usage Guide)

## 📝 개요 (Overview)

이 디렉토리에는 경기도 자율주행 프로젝트를 위한 커스텀 GitHub Copilot 에이전트 설정이 포함되어 있습니다.

This directory contains custom GitHub Copilot agent configurations for the Gyeonggi-do Autonomous Vehicle project.

## 🤖 사용 가능한 에이전트 (Available Agents)

### my-agent.agent.md
**경기도 자율주행 개발 도우미** - ROS2 기반 자율주행 시스템 개발 전문 에이전트

**주요 기능:**
- ROS2 Humble 패키지 개발 지원
- Python, OpenCV, YOLOv8 코드 작성 지원
- 커스텀 메시지 및 인터페이스 관리
- Arduino 하드웨어 통합 지원
- 빌드, 테스트, 디버깅 가이드
- 한글/영어 이중 언어 지원

## 🚀 사용 방법 (How to Use)

### GitHub Copilot Chat에서 사용

1. **에이전트 호출하기:**
   ```
   @my-agent 새로운 ROS2 노드를 만들어줘
   ```

2. **패키지별 질문하기:**
   ```
   @my-agent camera_pkg에서 YOLOv8 모델을 업데이트하려면?
   @my-agent lidar_pkg의 클러스터링 알고리즘을 개선하려면?
   @my-agent 새로운 커스텀 메시지를 interfaces_pkg에 추가하려면?
   ```

3. **문제 해결:**
   ```
   @my-agent 빌드 에러를 해결하려면?
   @my-agent 시리얼 통신이 안 될 때 어떻게 해?
   ```

4. **코드 리뷰 요청:**
   ```
   @my-agent 이 코드를 ROS2 베스트 프랙티스에 맞게 리뷰해줘
   ```

### GitHub Copilot CLI에서 사용

로컬 테스트를 위해 Copilot CLI 사용 가능:
```bash
# 에이전트 테스트
gh copilot explain "colcon build --symlink-install" --agent my-agent
```

## 📋 에이전트가 도움을 줄 수 있는 작업 (Tasks the Agent Can Help With)

- ✅ 새로운 ROS2 노드 생성
- ✅ Launch 파일 작성 및 수정
- ✅ 커스텀 메시지 정의 추가
- ✅ 컴퓨터 비전 코드 (OpenCV, YOLO) 작성
- ✅ 라이다 데이터 처리 알고리즘 개선
- ✅ 제어 알고리즘 개발 (조향, 속도)
- ✅ Arduino 통신 코드 작성
- ✅ 빌드 및 의존성 문제 해결
- ✅ 테스트 코드 작성
- ✅ 코드 최적화 및 리팩토링

## 🔧 에이전트 업데이트 (Updating the Agent)

에이전트 설정을 수정하려면 `my-agent.agent.md` 파일을 직접 편집하고 기본 브랜치에 머지하세요.

To update the agent configuration, edit the `my-agent.agent.md` file and merge it to the default branch.

변경 사항은 자동으로 GitHub Copilot에 반영됩니다.

Changes will be automatically reflected in GitHub Copilot.

## 📚 참고 자료 (References)

- [GitHub Copilot Custom Agents 문서](https://gh.io/customagents/config)
- [Copilot CLI 가이드](https://gh.io/customagents/cli)
- [프로젝트 README](../../README.md)

## 💡 팁 (Tips)

1. **구체적으로 질문하기**: "camera_pkg의 lane_detection.py에서 YOLOv8 신뢰도 임계값을 조정하려면?"
2. **컨텍스트 제공**: 현재 작업 중인 파일이나 코드를 포함하여 질문
3. **한글 사용 가능**: 에이전트는 한글과 영어 모두 이해합니다
4. **단계별 안내 요청**: "단계별로 설명해줘" 추가

---

Happy coding! 🚗✨
