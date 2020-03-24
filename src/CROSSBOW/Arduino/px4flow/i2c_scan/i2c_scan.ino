//test i2c device scanner from: https://playground.arduino.cc/Main/I2cScanner
#include <Wire.h>

void setup() {
  Wire.begin();
  Serial.begin(9600);
  while(!Serial);
  Serial.println("\nI2C Scanner");  
}

void loop() {
  byte error, address;
  int nDevices;

  Serial.println("Scanning...");
  nDevices=0;
  for(address=1;address<127;address++){
    Wire.beginTransmission(address);
    error=Wire.endTransmission();
    if (error==0){
      Serial.print("I2C device found at address 0x");
      if(address<16)
        Serial.print("0");
      Serial.println(address,HEX);
      delay(100);
      nDevices++;
    } //else {
      //Serial.print("Error ");
      //Serial.print(error);
      //Serial.print(" at address 0x");
      //Serial.println(address,HEX);
    //}
  }
  delay(1000);
}
