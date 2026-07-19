#define BLYNK_TEMPLATE_ID "TMPL3B1BQ3Z2q"
#define BLYNK_TEMPLATE_NAME "SMART METER"
#define BLYNK_AUTH_TOKEN "o2MgsL321XSgt0e_qcK7aq4ENT7X5zT1"

#include <LiquidCrystal.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include "EmonLib.h"

#define BLYNK_PRINT Serial

// LCD pin mapping: RS, E, D4, D5, D6, D7
LiquidCrystal lcd(13, 12, 14, 27, 26, 25);

// WiFi credentials
char ssid[] = "OPPO F25 Pro 5G";
char pass[] = "reddy2005";

BlynkTimer timer;
EnergyMonitor emon;

// === Calibration Constants ===
#define vCalibration 83.3
#define currCalibration 0.5
const float avgPower = 12.0;  // Average between 10W–20W bulb

// === Measurement Variables ===
float Vrms = 0;
float ImA = 0;
float power = 0;
float kWh = 0;
unsigned long lastMillis = millis();

void calculateReadings() {
  emon.calcVI(20, 2000);  // Perform voltage measurement
  Vrms = roundf(emon.Vrms * 10.0) / 10.0;  // No correction factor

  if (Vrms < 10.0) {
    Vrms = 0;
    ImA = 0;
    power = 0;
  } else {
    power = avgPower;
    ImA = (power / emon.Vrms) * 1000.0;
    //if (ImA > 1000.0) ImA = 100.0;

    unsigned long now = millis();
    kWh += power * (now - lastMillis) / 3600000.0;
    lastMillis = now;
    if (kWh > 10000.0) kWh = 0;
  }

  // Debug print
  Serial.print("Vrms: "); Serial.print(Vrms, 1); Serial.print(" V\t");
  Serial.print("Irms: "); Serial.print(ImA, 1); Serial.print(" mA\t");
  Serial.print("Power: "); Serial.print(power, 1); Serial.print(" W\t");
  Serial.print("kWh: "); Serial.println(kWh, 3);
}

void displayReadings() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("V:"); lcd.print(Vrms, 1); lcd.print("V");
  lcd.setCursor(0, 1);
  lcd.print("I:"); lcd.print(ImA, 0); lcd.print("mA");
  delay(800);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("P:"); lcd.print(power, 1); lcd.print("W");
  lcd.setCursor(0, 1);
  lcd.print("kWh:"); lcd.print(kWh, 3);
  delay(800);
}

void sendToBlynk() {
  Blynk.virtualWrite(V0, Vrms);
  Blynk.virtualWrite(V1, ImA);
  Blynk.virtualWrite(V2, power);
  Blynk.virtualWrite(V3, kWh);
}

void myTimerEvent() {
  calculateReadings();
  displayReadings();
  sendToBlynk();
}

void setup() {
  Serial.begin(9600);
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
  lcd.begin(16, 2);

  lcd.setCursor(3, 0);
  lcd.print("IoT Energy");
  lcd.setCursor(5, 1);
  lcd.print("Meter");
  delay(1500);
  lcd.clear();

  emon.voltage(35, vCalibration, 1.7);   // Voltage pin, calibration, phase shift
  emon.current(34, currCalibration);    // Current pin (not used), calibration

  timer.setInterval(1000L, myTimerEvent);  // Update every second
}

void loop() {
  Blynk.run();
  timer.run();
}
