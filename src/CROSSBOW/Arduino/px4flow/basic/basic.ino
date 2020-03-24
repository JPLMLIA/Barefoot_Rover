/*
 * Copyright (c) 2014 by Laurent Eschenauer <laurent@eschenauer.be>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 */

#include <Wire.h>
#include "PX4Flow.h"

PX4Flow sensor = PX4Flow(); 

void setup()
{
  Wire.begin();       
  Serial.begin(115200);  
}

void loop()
{
  if(!sensor.update()){
    Serial.println("error");
  }
  
  Serial.print("#");
  Serial.print(sensor.frame_count());Serial.print(",");
  Serial.print(sensor.pixel_flow_x_sum());Serial.print(",");
  Serial.print(sensor.pixel_flow_y_sum());Serial.print(",");
  Serial.print(sensor.flow_comp_m_x());Serial.print(",");
  Serial.print(sensor.flow_comp_m_y());Serial.print(",");
  Serial.print(sensor.sonar_timestamp());Serial.print(",");
  Serial.println(sensor.ground_distance());

  delay(100);
}

