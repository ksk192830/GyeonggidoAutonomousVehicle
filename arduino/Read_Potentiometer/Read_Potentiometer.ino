int potPin = A0;   // 가변저항 가운데 핀을 A0에 연결
int potValue = 0;

void setup() {
  Serial.begin(115200); // 시리얼 통신 시작
}

void loop() {
  potValue = analogRead(potPin); // 0~1023 값 읽기
  Serial.println(potValue);      // 읽은 값 출력
  delay(100); // 너무 빠르게 출력되는거 방지
}
