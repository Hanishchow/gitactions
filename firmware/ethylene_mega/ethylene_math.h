// Sensor formulas, mirroring ethylene/physics.py (the single source of truth):
//   c      = A * (Rs/R0)^B
//   f_T    = sqrt(303.15 / (T + 273.15))
//   f_RH   = 1 / (1 + 0.15 * RH / 100)
//   c_comp = c * f_T * f_RH
// Invalid inputs return NaN, never inf. On AVR double == float (32-bit).
#ifndef ETHYLENE_MATH_H
#define ETHYLENE_MATH_H

#include <math.h>
#include "config.h"

#define KELVIN 273.15f
#define T_REF_K 303.15f
#define RH_COEFF 0.15f

// Sensor resistance (ohm) from a (possibly averaged) ADC reading; NaN on either rail.
static inline float adc_to_rs(float adc) {
  if (!(adc > 0.0f) || adc >= (float)ADC_MAX) return NAN;
  return (float)RL_OHM * ((float)ADC_MAX - adc) / adc;
}

// Concentration from Rs/R0 via the power law; NaN if the ratio is not positive (or NaN).
static inline float concentration(float rs_r0) {
  if (!(rs_r0 > 0.0f)) return NAN;
  return (float)SENSOR_A * powf(rs_r0, (float)SENSOR_B);
}

// Temperature correction factor; NaN at or below absolute zero.
static inline float f_temp(float temp_c) {
  if (!(temp_c > -KELVIN)) return NAN;
  return sqrtf(T_REF_K / (temp_c + KELVIN));
}

// Relative-humidity correction factor; NaN if the denominator is not positive.
static inline float f_rh(float rh) {
  float denom = 1.0f + RH_COEFF * rh / 100.0f;
  if (!(denom > 0.0f)) return NAN;
  return 1.0f / denom;
}

static inline float compensate(float c, float temp_c, float rh) {
  return c * f_temp(temp_c) * f_rh(rh);
}

// Mirrors physics.dht_in_range: DHT11 valid range.
static inline bool dht_in_range(float temp_c, float rh) {
  return temp_c >= 0.0f && temp_c <= 50.0f && rh >= 0.0f && rh <= 100.0f;
}

#endif
