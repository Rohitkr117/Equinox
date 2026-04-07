#include <Wire.h>

const int MPU_ADDR = 0x68; 
int16_t ax, ay, az, gx, gy, gz;

void setup() {
  Serial.begin(115200);
  delay(2000); // USB stability delay

  // SuperMini / ESP32C3 specific I2C pins
  Wire.begin(2, 3); 

  // Wake up the MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); 
  Wire.write(0);
  Wire.endTransmission(true);

  Serial.println("MPU6050 Raw Stream Initialized.");
}

void loop() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true); 
  
  if (Wire.available() == 14) {
    ax = Wire.read()<<8 | Wire.read(); 
    ay = Wire.read()<<8 | Wire.read(); 
    az = Wire.read()<<8 | Wire.read(); 
    Wire.read()<<8 | Wire.read(); // Skip temperature
    gx = Wire.read()<<8 | Wire.read(); 
    gy = Wire.read()<<8 | Wire.read(); 
    gz = Wire.read()<<8 | Wire.read(); 

    // Stream the raw values over Serial as a comma-separated string
    Serial.print(ax); Serial.print(",");
    Serial.print(ay); Serial.print(",");
    Serial.print(az); Serial.print(",");
    Serial.print(gx); Serial.print(",");
    Serial.print(gy); Serial.print(",");
    Serial.println(gz); // newline marks the end of the packet
  } else {
    // If the sensor fails, gracefully attempt to restart the I2C connection
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B); 
    Wire.write(0);
    Wire.endTransmission(true);
  }

  // 20-millisecond delay for approximately 50Hz sample rate
  delay(20); 
}
