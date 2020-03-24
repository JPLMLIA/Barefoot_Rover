//simplified version of px4flow pose_estimate example to return pixel changes
  //ignore gyro effects (Barefoot cart should remain aligned to something consistently)
#include <Wire.h>
#include "PX4Flow.h"

// Initialize PX4Flow library
PX4Flow sensor = PX4Flow();

float px = 0;
float py = 0;
long last_check=0;

void setup() {
  Wire.begin();
  Serial.begin(115200);
}

void loop() {
  long loop_start = millis(); //ms
  float dur = (loop_start-last_check) / 1000.0f; //sec
  
  sensor.update(); //basic (not integral) values

  if (sensor.qual()>100){ //0-255 metric for how the images turn out
    px = px + sensor.pixel_flow_x_sum() * dur * 10.0f; //pixels*100
    py = py + sensor.pixel_flow_y_sum() * dur * 10.0f;
  }
  Serial.print(millis()); Serial.print(",");
  Serial.print(sensor.frame_count()); Serial.print(",");
  Serial.print(px); Serial.print(",");
  Serial.print(py); Serial.println("");
  last_check = loop_start;

  delay(100);
}
