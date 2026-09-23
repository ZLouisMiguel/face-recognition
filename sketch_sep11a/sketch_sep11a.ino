#include <Servo.h>

Servo lockServo;

void setup() {
  Serial.begin(9600);
  lockServo.attach(9);
  lockServo.write(0); // Default locked position
}

void loop() {
  if (Serial.available() > 0) {
    char signal = Serial.read();
    if (signal == '1') {
      lockServo.write(90); // Unlock position
      delay(3000);
      lockServo.write(0);  // Re-lock
    }
  }
}