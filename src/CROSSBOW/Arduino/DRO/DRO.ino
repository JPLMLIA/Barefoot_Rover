/*
based off of yuriystoys documentation: http://www.yuriystoys.com/2012/01/reading-gtizzly-igaging-scales-with.html
- only need single data pin, clock pin (both digital)
- clock pin outputs 5V, reduced to 3V w/ voltage divider
- input tied to ground
*/

//avoid tx,rx pins
int const clockPin = 2;
int const dataPin = 3;

int bitOffset;
long coord;

void setup() {
  Serial.begin(9600);
  pinMode(clockPin,OUTPUT);
  pinMode(dataPin,INPUT);
}

void loop() {
  coord = 0;  //reset
  for (bitOffset=0;bitOffset<20;bitOffset++){
    digitalWrite(clockPin,HIGH);  //tick
    __asm__("nop\n\t");
    __asm__("nop\n\t");
    __asm__("nop\n\t");
    digitalWrite(clockPin,LOW); //tock
    coord |= (digitalRead(dataPin)<<bitOffset);
  }
  
  //sign correction:
  digitalWrite(clockPin,HIGH);
  __asm__("nop\n\t");
  __asm__("nop\n\t");
  __asm__("nop\n\t");
  digitalWrite(clockPin,LOW);
  if (digitalRead(dataPin)==HIGH){
    coord |= (0x7ff << 21);
  }
  Serial.println(coord);
}
