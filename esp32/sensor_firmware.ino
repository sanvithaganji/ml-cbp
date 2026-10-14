/*
  ==============================================================================
  ESP32 Firmware: Microclimate Environmental Node for Crop Disease Early Warning
  ==============================================================================
  Target Board: ESP32 Dev Module (WROOM-32)
  Sensors Connected:
    1. DHT22 (AM2302)                  -> Ambient Temp & Relative Humidity (Pin 4)
    2. DS18B20 (Dallas OneWire)        -> Soil Temperature (Pin 15)
    3. Capacitive Soil Moisture v1.2   -> Soil Moisture % (Pin 34 - ADC1_CH6)
    4. Leaf Wetness Grid (Resistive)   -> Leaf Wetness Duration / Index (Pin 35 - ADC1_CH7)

  Features:
    - Wi-Fi connectivity with auto-reconnect
    - JSON payload formatting with timestamps & hardware UUID
    - Sensor disconnection & out-of-range detection (transmits null if disconnected)
    - Configurable telemetry sampling interval (default 15 seconds)
    - HTTP POST transmission to local server endpoint: /api/telemetry
  ==============================================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ArduinoJson.h>

// -----------------------------------------------------------------------------
// 1. Configuration & Network Credentials
// -----------------------------------------------------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Target Receiver URL (Update IP to your Mac / Server running sensor_receiver.py)
const char* SERVER_ENDPOINT = "http://192.168.1.100:8080/api/telemetry";

// Telemetry interval in milliseconds (e.g., 15000 ms = 15 seconds)
const unsigned long SAMPLING_INTERVAL_MS = 15000;
unsigned long lastSampleTime = 0;

// -----------------------------------------------------------------------------
// 2. Hardware Pin Definitions & Sensor Objects
// -----------------------------------------------------------------------------
#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

#define ONE_WIRE_BUS 15
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature soilTempSensor(&oneWire);

#define SOIL_MOISTURE_PIN 34  // Analog Input ADC1
#define LEAF_WETNESS_PIN  35  // Analog Input ADC1

// Calibration constants for Capacitive Soil Moisture v1.2 (12-bit ADC: 0 - 4095)
// Air reading (0% moisture): ~3200 | Water submerged (100% moisture): ~1400
const int SOIL_AIR_RAW   = 3200;
const int SOIL_WATER_RAW = 1400;

// Calibration constants for Leaf Wetness Grid (0 - 4095)
// Completely dry leaf: ~4000 | Wet leaf droplet bridge: < 1800
const int LEAF_DRY_RAW = 3800;
const int LEAF_WET_RAW = 1600;

// -----------------------------------------------------------------------------
// 3. Setup
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[ESP32] Initializing Multimodal Microclimate Node...");

  // Initialize sensors
  dht.begin();
  soilTempSensor.begin();
  pinMode(SOIL_MOISTURE_PIN, INPUT);
  pinMode(LEAF_WETNESS_PIN, INPUT);

  // Connect to Wi-Fi
  connectWiFi();
}

// -----------------------------------------------------------------------------
// 4. Main Loop
// -----------------------------------------------------------------------------
void loop() {
  // Ensure Wi-Fi remains active
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  unsigned long currentMillis = millis();
  if (currentMillis - lastSampleTime >= SAMPLING_INTERVAL_MS) {
    lastSampleTime = currentMillis;
    readAndTransmitTelemetry();
  }
}

// -----------------------------------------------------------------------------
// 5. Sensor Acquisition & HTTP Transmission
// -----------------------------------------------------------------------------
void readAndTransmitTelemetry() {
  // 1. Ambient Temp & Humidity
  float ambientTemp = dht.readTemperature();
  float humidity    = dht.readHumidity();

  // 2. Soil Temperature
  soilTempSensor.requestTemperatures();
  float soilTemp = soilTempSensor.getTempCByIndex(0);
  bool soilTempValid = (soilTemp > -55.0 && soilTemp < 85.0 && soilTemp != DEVICE_DISCONNECTED_C);

  // 3. Soil Moisture
  int rawSoil = analogRead(SOIL_MOISTURE_PIN);
  float soilMoisturePercent = 0.0;
  bool soilMoistureValid = (rawSoil > 200 && rawSoil < 4095); // disconnected check
  if (soilMoistureValid) {
    soilMoisturePercent = map(rawSoil, SOIL_AIR_RAW, SOIL_WATER_RAW, 0, 100);
    soilMoisturePercent = constrain(soilMoisturePercent, 0.0, 100.0);
  }

  // 4. Leaf Wetness Duration / Index
  int rawLeaf = analogRead(LEAF_WETNESS_PIN);
  float leafWetnessHours = 0.0; // In field deployment, accumulated hours
  bool leafWetnessValid = (rawLeaf > 100 && rawLeaf < 4095);
  if (leafWetnessValid) {
    float wetnessIndex = map(rawLeaf, LEAF_DRY_RAW, LEAF_WET_RAW, 0, 100);
    wetnessIndex = constrain(wetnessIndex, 0.0, 100.0);
    // Approximation: 100% surface saturation maps to active leaf wetness
    leafWetnessHours = (wetnessIndex / 100.0) * 12.0; 
  }

  // Build JSON Document
  StaticJsonDocument<512> doc;
  doc["device_id"]   = WiFi.macAddress();
  doc["uptime_sec"]  = millis() / 1000;
  doc["hardware_mode"] = "PHYSICAL_ESP32";

  JsonObject telemetry = doc.createNestedObject("telemetry");

  // Populate or send null if sensor is disconnected
  if (!isnan(ambientTemp) && ambientTemp >= -10.0 && ambientTemp <= 60.0) {
    telemetry["ambient_temperature"] = ambientTemp;
  } else {
    telemetry["ambient_temperature"] = nullptr;
    Serial.println("[WARN] DHT22 Temperature disconnected or invalid.");
  }

  if (!isnan(humidity) && humidity >= 0.0 && humidity <= 100.0) {
    telemetry["humidity"] = humidity;
  } else {
    telemetry["humidity"] = nullptr;
    Serial.println("[WARN] DHT22 Humidity disconnected or invalid.");
  }

  if (soilTempValid) {
    telemetry["soil_temperature"] = soilTemp;
  } else {
    telemetry["soil_temperature"] = nullptr;
    Serial.println("[WARN] DS18B20 Soil Temp disconnected.");
  }

  if (soilMoistureValid) {
    telemetry["soil_moisture"] = soilMoisturePercent;
  } else {
    telemetry["soil_moisture"] = nullptr;
    Serial.println("[WARN] Soil Moisture sensor disconnected.");
  }

  if (leafWetnessValid) {
    telemetry["leaf_wetness"] = leafWetnessHours;
  } else {
    telemetry["leaf_wetness"] = nullptr;
    Serial.println("[WARN] Leaf Wetness sensor disconnected / unavailable.");
  }

  String jsonPayload;
  serializeJson(doc, jsonPayload);
  Serial.println("[ESP32] Telemetry: " + jsonPayload);

  // Send HTTP POST
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER_ENDPOINT);
    http.addHeader("Content-Type", "application/json");

    int httpCode = http.POST(jsonPayload);
    if (httpCode > 0) {
      Serial.printf("[ESP32] POST response code: %d\n", httpCode);
    } else {
      Serial.printf("[ESP32] POST failed, error: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
  }
}

// -----------------------------------------------------------------------------
// 6. Wi-Fi Helper
// -----------------------------------------------------------------------------
void connectWiFi() {
  Serial.print("[ESP32] Connecting to Wi-Fi SSID: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[ESP32] Wi-Fi Connected!");
    Serial.print("[ESP32] Local IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[ESP32] Wi-Fi connection timed out. Will retry.");
  }
}
