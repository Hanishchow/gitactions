// DHT11 bit-banged driver, no external library.
//
// Protocol: host pulls the line low >= 18 ms, releases it; the sensor answers 80 us low + 80 us high,
// then sends 40 bits, each 50 us low followed by 26-28 us high (0) or 70 us high (1).
// Bytes: RH int, RH dec, T int, T dec, checksum (low byte of the sum of the first four).
//
// Interrupts are disabled only for the ~4 ms handshake + 40-bit read. During that window micros()
// cannot be trusted (Timer0 overflows are not serviced, so it jumps back by 1024 us once a second
// overflow is pending), so pulse widths and timeouts are measured in busy-loop iterations instead:
// a bit is 1 when its high phase lasts longer than its 50 us low phase, which needs no calibration.
// Every wait is bounded by DHT_TIMEOUT_LOOPS (about 1 ms), so a missing sensor costs ~21 ms, never a hang.
// Callers must leave >= 1 s between reads (the sketch reads once per SAMPLE_MS = 2 s).
#ifndef ETHYLENE_DHT11_H
#define ETHYLENE_DHT11_H

#include <Arduino.h>

#define DHT_TIMEOUT_LOOPS ((uint16_t)(F_CPU / 8000UL))  // ~8 cycles per loop -> about 1 ms
#define DHT_TIMEOUT 0xFFFF

// Busy-waits while the pin stays at `level`; returns loop count, or DHT_TIMEOUT.
static inline uint16_t dht_pulse(volatile uint8_t *in, uint8_t mask, uint8_t level) {
  uint16_t count = 0;
  uint8_t want = level ? mask : 0;
  while ((*in & mask) == want) {
    if (++count >= DHT_TIMEOUT_LOOPS) return DHT_TIMEOUT;
  }
  return count;
}

// Reads one frame. True only with a complete frame and a valid checksum; outputs untouched otherwise.
// The caller applies the 0-50 degC / 0-100 %RH range check (dht_in_range in ethylene_math.h).
static bool dht11_read(uint8_t pin, int8_t *temp_c, uint8_t *rh) {
  uint8_t mask = digitalPinToBitMask(pin);
  volatile uint8_t *in = portInputRegister(digitalPinToPort(pin));
  uint8_t data[5] = {0, 0, 0, 0, 0};

  // Start signal: >= 18 ms low (interrupts stay enabled for this part).
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
  delay(20);

  bool ok = true;
  uint8_t sreg = SREG;
  noInterrupts();
  pinMode(pin, INPUT_PULLUP);  // release; the sensor pulls low 20-40 us later
  delayMicroseconds(10);
  // Wait out the rest of the host-high gap, then the sensor's 80 us low and 80 us high.
  if (dht_pulse(in, mask, 1) == DHT_TIMEOUT || dht_pulse(in, mask, 0) == DHT_TIMEOUT ||
      dht_pulse(in, mask, 1) == DHT_TIMEOUT) {
    ok = false;
  }
  for (uint8_t i = 0; ok && i < 40; i++) {
    uint16_t low = dht_pulse(in, mask, 0);
    uint16_t high = dht_pulse(in, mask, 1);
    if (low == DHT_TIMEOUT || high == DHT_TIMEOUT) {
      ok = false;
    } else {
      data[i >> 3] = (uint8_t)((data[i >> 3] << 1) | (high > low ? 1 : 0));
    }
  }
  SREG = sreg;  // restore the interrupt state

  if (!ok) return false;
  if ((uint8_t)(data[0] + data[1] + data[2] + data[3]) != data[4]) return false;
  // An all-zero frame passes the checksum but is what a glitching line produces; reject it.
  if ((data[0] | data[1] | data[2] | data[3]) == 0) return false;

  // Some DHT11 revisions flag sub-zero temperatures in bit 7 of the decimal byte.
  int16_t t = (int16_t)(data[2] & 0x7F);
  if (data[3] & 0x80) t = -t;
  *temp_c = (int8_t)t;
  *rh = data[0];
  return true;
}

#endif
