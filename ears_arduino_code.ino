///// Sensor Define /////
const int leftSoundSensor = A5;
const int rightSoundSensor = A0;

///// Left Sensor Variables /////
int leftAverageCounter = 0;
int leftAverage = 0;
int leftBorder = 0;

///// Right Sensor Variables /////
int rightAverageCounter = 0;
int rightAverage = 0;
int rightBorder = 0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  leftAverageCounter++;
  rightAverageCounter++;
  
  long leftSum = 0;
  long rightSum = 0;
  for (int i = 0; i < 32; i++) {
    leftSum += analogRead(leftSoundSensor);
  }
  for (int i = 0; i < 32; i++) {
    rightSum += analogRead(rightSoundSensor);
  }

  leftSum >>= 5;
  rightSum >>= 5;

  if (leftAverageCounter >= 9 && leftAverageCounter <= 13) {
    leftAverage = leftAverage + leftSum;
  }
  if (rightAverageCounter >= 9 && rightAverageCounter <= 13) {
    rightAverage = rightAverage + rightSum;
  }

  leftBorder = (leftAverage/5) + 50;
  rightBorder = (rightAverage/5) + 50;
  if (leftSum > leftBorder && leftAverageCounter >= 13 && leftSum > rightSum) {
    Serial.println("Turn Left!");
  }
  if (rightSum > rightBorder && rightAverageCounter >= 13 && leftSum < rightSum) {
    Serial.println("Turn Right!");
  }
  delay(300);
}
