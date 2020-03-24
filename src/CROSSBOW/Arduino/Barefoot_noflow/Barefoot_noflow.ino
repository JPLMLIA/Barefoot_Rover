/*
Combination Arduino script to handle data from DRO, ADXL345 accelerometer, ESCON controller, px4flow (removed)
Created: 2018-01-18
Last Modified: 2018-11-29 (to remove flow sensor w/o changing data stream output)
Author: Raymond Ma (153268)
*/
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
//#include "PX4Flow.h"

//int const msDelay = 20; //help determine data rate (50Hz)
long atime; //millis timing
long last_atime = 0;

//DRO pins (DO2,DI3)
int const clockPin = 2;
int const dataPin = 3;
int bitOffset;
long coord; //corresponds to 0.01mm

//ESCON (analog 0,1)
int speedVal; //0-1024 value, w/ limits set in ESCON Studio
int currVal;  

//ADXL345 (SDA,SCL) 0x53
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(54321);
float ax;   //RT acceleration x
float ay;   //RT acceleration y
float az;   //RT acceleration z

//PX4Flow (SDA,SCL) 0x42
//PX4Flow sensor = PX4Flow();
//float px = 0;
//float py = 0;

void setup() {
  Wire.begin();  //for both ADXL and px4flow
  Serial.begin(115200);

  //DRO:
  pinMode(clockPin,OUTPUT);
  pinMode(dataPin,INPUT);

  //IMU:
  delay(500);	//maybe timing is source of IMU startup issues...
  if(!accel.begin())
  {
    /* There was a problem detecting the ADXL345 ... check your connections */
    Serial.println("[ERR-Cart] Arduino failed to detect ADXL345");
    while(1);	//forces timeout error on computer side
  }
  
  /* Set the range to whatever is appropriate for your project */
  //accel.setRange(ADXL345_RANGE_2_G);
}

void loop() {
  atime = millis();   //ms from startup (or last Serial monitor connection)
  float dur = atime-last_atime; //ms, tracked even if not polled
  
  if (Serial.available() > 0 && Serial.read()=='R'){
    
    //px4flow
    //sensor.update();  //WARNING: no quality check qualifier due to other measured parameters (integral not updated)
    //px = px + sensor.pixel_flow_x_sum() * dur * 10.0f; //pixels*100
    //py = py + sensor.pixel_flow_y_sum() * dur * 10.0f;
  
    //IMU
    sensors_event_t event; 
    accel.getEvent(&event);
  
    //DRO
    coord=0;
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

    //ESCON
    speedVal = analogRead(0);
    currVal = analogRead(1);

    Serial.print(atime); Serial.print("\t");              //ms timing 
    Serial.print(coord); Serial.print("\t");              //z-stage value (0.01mm)
    Serial.print(currVal); Serial.print("\t");            //ESCON current
    Serial.print(speedVal); Serial.print("\t");           //ESCON speed
    Serial.print(event.acceleration.x); Serial.print("\t"); //ADXL x
    Serial.print(event.acceleration.y); Serial.print("\t"); //ADXL y
    Serial.print(event.acceleration.z); Serial.print("\t"); //ADXL z
    Serial.print(0); Serial.print("\t");                   //px4flow x displacement
    Serial.print(0);                                       //px4flow y displacement (pixel * 100)
    Serial.println("");
    Serial.flush();	//makes sure print buffer is sent before continuing
    last_atime = atime;
    while(Serial.available()>0){
      Serial.read();  //clear read buffer
    }
  }
}
