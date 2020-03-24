//Risaku's code for ADXL from Tactile Wheel implementation
//only update for Barefoot is change of baudrate from 9600 to 115200

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

/* Assign a unique ID to this sensor at the same time */
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

int inByte = 0;  //RT

float ax;   //RT acceleration x
float ay;   //RT acceleration y
float az;   //RT acceleration z

void setup(void) 
{
  Serial.begin(115200);

  /* Initialise the sensor */
  if(!accel.begin())
  {
    /* There was a problem detecting the ADXL345 ... check your connections */
    Serial.println("[ERR][Wheel] Arduino failed to detect ADXL345");
    while(1);	//forces timeout error on computer side
  } 
  accel.setRange(ADXL345_RANGE_2_G);
}

void loop(void) 
{
  while (Serial.available() == 0)   //RT
  {   //RT
  }   //RT
  
  if (Serial.available() > 0);  //RT
  {
   inByte = Serial.read();  //RT

   if (inByte == 'R'){    //RT
      /* Get a new sensor event regardless of user query*/ 
      sensors_event_t event; 
      accel.getEvent(&event);
     /* Display the results (acceleration is measured in m/s^2) */
      Serial.print(event.acceleration.x); Serial.print("\t");
      Serial.print(event.acceleration.y); Serial.print("\t");
      Serial.print(event.acceleration.z); Serial.println("");
	  Serial.flush();
   }
}
}
