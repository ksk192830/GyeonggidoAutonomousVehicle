// 핀 번호 변수
const int Motor_1 = 2;
const int Motor_2 = 3;

void setup() {
  // put your setup code here, to run once:
  pinMode(Motor_1, OUTPUT);
  pinMode(Motor_2, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  analogWrite(Motor_1, 100);
  analogWrite(Motor_2, LOW);
}
