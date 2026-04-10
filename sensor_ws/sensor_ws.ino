#include <Wire.h>
#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ESPmDNS.h>

// Forward declaration to prevent Arduino IDE preprocessor issues with WStype_t
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length);

// =========================================================================
// 1. Wi-Fi STATION SETTINGS (Connect to Laptop Hotspot)
//    The ESP32 connects as a client, saving massive battery & heat.
// =========================================================================
const char* wifi_ssid     = "Equinox-Sensor";
const char* wifi_password = "equinox123";

// =========================================================================
// 2. SENSOR SETTINGS
// =========================================================================
const int MPU_ADDR = 0x68;
int16_t ax, ay, az, gx, gy, gz;

// =========================================================================
// 3. WEBSOCKET SERVER (runs on the ESP32 itself, port 81)
// =========================================================================
WebSocketsServer webSocket = WebSocketsServer(81);
bool clientConnected = false;

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      clientConnected = false;
      Serial.printf("[WS] Client #%u disconnected.\n", num);
      break;
    case WStype_CONNECTED:
      clientConnected = true;
      Serial.printf("[WS] Client #%u connected from %s\n", num, webSocket.remoteIP(num).toString().c_str());
      break;
    default:
      break;
  }
}

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

  // --- High-Sensitivity Tremor Detection Configuration ---

  // 1. Enable DLPF at 21 Hz bandwidth (reduces noise by 3.5x)
  //    Noise drops from ~6.45 mg RMS (260 Hz BW) to ~1.83 mg RMS
  //    Passes the full 4-12 Hz tremor band while rejecting high-freq noise
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1A);  // CONFIG register
  Wire.write(0x04);  // DLPF_CFG = 4 → Accel BW = 21 Hz, Gyro BW = 20 Hz
  Wire.endTransmission(true);

  // 2. Set sample rate: 1 kHz / (1 + 19) = 50 Hz (hardware-enforced)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x19);  // SMPLRT_DIV register
  Wire.write(19);    // Divider = 19 → 50 Hz output
  Wire.endTransmission(true);

  // 3. Accelerometer at ±2g (most sensitive: 16384 LSB/g)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);  // ACCEL_CONFIG register
  Wire.write(0x00);  // AFS_SEL = 0 → ±2g
  Wire.endTransmission(true);

  // 4. Gyroscope at ±250°/s (most sensitive: 131 LSB/°/s)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);  // GYRO_CONFIG register
  Wire.write(0x00);  // FS_SEL = 0 → ±250°/s
  Wire.endTransmission(true);

  // -----------------------------------------------------------------------
  // Start the ESP32 as a Wi-Fi Client (Station Mode)
  // -----------------------------------------------------------------------
  WiFi.mode(WIFI_STA);
  WiFi.begin(wifi_ssid, wifi_password);
  
  Serial.print("\nConnecting to Wi-Fi: ");
  Serial.println(wifi_ssid);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n--- Connected to Wi-Fi ---");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  
  // Start mDNS Responder: equinox.local
  if (!MDNS.begin("equinox")) {
    Serial.println("Error setting up MDNS responder!");
    Serial.println("mDNS responder started! Python can connect to ws://equinox.local:81");
  }

  // Start the WebSocket server on port 81
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);

  Serial.println("MPU6050 WebSocket Stream Initialized. Waiting for client...");
}

void loop() {
  webSocket.loop(); // Must be called every iteration to handle connections

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

    // Stream the raw values over WebSocket as a comma-separated string
    // Same exact format as the Serial version: ax,ay,az,gx,gy,gz
    if (clientConnected) {
      String packet = String(ax) + "," + String(ay) + "," + String(az) + "," +
                      String(gx) + "," + String(gy) + "," + String(gz);
      webSocket.broadcastTXT(packet);
    }
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
