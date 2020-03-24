/*Simple analog read to evaluate the Maxon ESCON speed and current outputs*/

void setup() {
  Serial.begin(9600);
}
int speedVal;
int currVal;
void loop() {
  speedVal = analogRead(0); //0-1024 value, depending on range set in ESCON Studio
  currVal = analogRead(1);

  Serial.print(speedVal); Serial.print("  ");
  Serial.print(currVal); Serial.println("");
}
