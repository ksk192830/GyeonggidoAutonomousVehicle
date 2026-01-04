// angle 수정하여 조향 조절
const int STEERING_1 = 6;
const int STEERING_2 = 7;
const int POT = A0;

// 가변저항 값 범위
const int resistance_most_left = 620; 
const int resistance_most_right = 450;

// 조향 최대 단계 수 (한 쪽 기준)
const int MAX_STEERING_STEP = 10;

// 조향 속도 상수
const int STEERING_SPEED = 150;

// 제어 상태 변수 -- angle에 -10 ~ 10 입력해서 업로드
int angle = 0, resistance = 0, mapped_resistance = 0;

// 함수 선언
void steerRight();
void steerLeft();

void setup() {
  // 핀 모드 설정
    pinMode(POT, INPUT);
    pinMode(STEERING_1, OUTPUT);
    pinMode(STEERING_2, OUTPUT);
}

void loop() {
// 포텐셔미터 값을 읽어 조향 계산
  resistance = analogRead(POT);
  mapped_resistance = map(resistance, resistance_most_left, resistance_most_right, -MAX_STEERING_STEP, MAX_STEERING_STEP + 1);

  // 조향 상태에 따라 동작 제어
  if (mapped_resistance == angle) {
      maintainSteering();
  } else if (mapped_resistance > angle) {
      steerLeft();
  } else {
      steerRight();
  }

}

// 조향 제어 함수
void steerRight() {
    analogWrite(STEERING_1, STEERING_SPEED);
    analogWrite(STEERING_2, LOW);
}

void steerLeft() {
    analogWrite(STEERING_1, LOW);
    analogWrite(STEERING_2, STEERING_SPEED);
}

void maintainSteering() {
    analogWrite(STEERING_1, LOW);
    analogWrite(STEERING_2, LOW);
}
