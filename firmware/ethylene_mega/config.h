// Ethylene instrument, Arduino Mega 2560: pins, sensor constants, timing, flags.
// Constants mirrored from ethylene/physics.py and ethylene/data.py; tests/test_firmware_parity.py
// checks they still match.
#ifndef ETHYLENE_CONFIG_H
#define ETHYLENE_CONFIG_H

// ---- Pins ----
#define PIN_MQ A0         // MQ-135 analog out (load resistor RL to GND)
#define PIN_LDR A1        // LDR divider
#define PIN_DHT 22        // DHT11 data (needs a 4.7k-10k pull-up to 5 V)
#define PIN_VENT 30       // vent relay
#define PIN_SCRUB 31      // scrubber relay
#define PIN_LED 13        // status LED (on = warm-up / baseline)
#define SERIAL_BAUD 115200

// Relay modules differ: 1 = energise with HIGH, 0 = active-low module.
#define RELAY_ACTIVE_HIGH 1

// ---- Sensor model (paper / common MQ-135 curve) ----
#define SENSOR_A 116.6020682
#define SENSOR_B -2.769034857
#define RL_OHM 10000.0
#define ADC_MAX 1023
// Each MQ and LDR value is the mean of ADC_AVG analogRead()s about 1 ms apart (noise reduction).
#define ADC_AVG 8

// ---- Timing (ms) ----
#define WARMUP_MS 180000UL    // heater warm-up after power-up, no index
#define BASELINE_MS 30000UL   // last part of warm-up: R0 = mean Rs, LDR_blk = mean LDR (clean air)
#define SAMPLE_MS 2000UL      // one CSV line per sample; also keeps DHT11 reads >= 1 s apart

// ---- Relay thresholds on the ripeness index, with hysteresis ----
#define VENT_IDX 0.90
#define SCRUB_IDX 0.85
#define RELAY_HYST 0.03

// ---- CSV flags bit field ----
#define FLAG_WARMUP 1
#define FLAG_DHT_FAIL 2
#define FLAG_MQ_FAULT 4
#define FLAG_VENT 8
#define FLAG_SCRUB 16

#endif
