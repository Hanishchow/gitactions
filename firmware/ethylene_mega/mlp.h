// The paper's 6-256-128-64-32-1 MLP, int8 weights with per-row float scales, read from PROGMEM.
// Layout and forward pass: contracts section A (mirrored in Python by ethylene/export.py), except that
// W2 is split into W2A (rows 0-63) and W2B (rows 64-127) because avr-gcc caps one object at 32767 bytes:
//   a0 = (x - SCALER_MEAN) / SCALER_STD
//   a_out[j] = (sum_i W[j*in + i] * a_in[i]) * S[j] + B[j]      ReLU after layers 1-4, none after 5.
// Activations live in two 256-float ping-pong buffers (2 KB SRAM total), nothing else is allocated.
//
// pgm_read_byte/pgm_read_float use 16-bit (near) addresses: fine while the PROGMEM tables stay in the
// low 64 KB of flash. avr-gcc places .progmem right after the vector table, and this model's tables are
// about 48 KB, so they do. A model with more than ~60 KB of tables would need pgm_read_*_far.
#ifndef ETHYLENE_MLP_H
#define ETHYLENE_MLP_H

#include <avr/pgmspace.h>
// Arduino's binary.h defines B1 (= 1) for binary literals; it would clash with the bias table B1.
#ifdef B1
#undef B1
#endif
#include "model_weights.h"

#define MLP_BUF 256
static float mlp_buf_a[MLP_BUF];
static float mlp_buf_b[MLP_BUF];

static void mlp_layer(const int8_t *W, const float *S, const float *B, const float *in, uint16_t n_in,
                      float *out, uint16_t n_out, bool relu) {
  for (uint16_t j = 0; j < n_out; j++) {
    const int8_t *row = W + (uint16_t)(j * n_in);
    float acc = 0.0f;
    for (uint16_t i = 0; i < n_in; i++) {
      float a = in[i];
      if (a == 0.0f) continue;  // exact skip: after ReLU many activations are 0, saves the soft-float MAC
      acc += (float)(int8_t)pgm_read_byte(row + i) * a;
    }
    float y = acc * pgm_read_float(S + j) + pgm_read_float(B + j);
    if (relu && y < 0.0f) y = 0.0f;
    out[j] = y;
  }
}

// x in model input order [Rs/R0, c_comp, LDR_raw, LDR_blk, T, RH]. Returns the raw (unclamped) index;
// non-finite when the weights are unusable (e.g. the zero-filled placeholder) -- callers must check.
float mlp_forward(const float x[MLP_IN]) {
  for (uint8_t i = 0; i < MLP_IN; i++) {
    mlp_buf_a[i] = (x[i] - pgm_read_float(SCALER_MEAN + i)) / pgm_read_float(SCALER_STD + i);
  }
  mlp_layer(W1, S1, B1, mlp_buf_a, MLP_IN, mlp_buf_b, MLP_L1, true);
  // W2 (128x256 = 32 KB) is stored as two 64-row halves: no AVR object may reach 32768 bytes.
  mlp_layer(W2A, S2, B2, mlp_buf_b, MLP_L1, mlp_buf_a, MLP_L2 / 2, true);
  mlp_layer(W2B, S2 + MLP_L2 / 2, B2 + MLP_L2 / 2, mlp_buf_b, MLP_L1, mlp_buf_a + MLP_L2 / 2, MLP_L2 / 2, true);
  mlp_layer(W3, S3, B3, mlp_buf_a, MLP_L2, mlp_buf_b, MLP_L3, true);
  mlp_layer(W4, S4, B4, mlp_buf_b, MLP_L3, mlp_buf_a, MLP_L4, true);
  mlp_layer(W5, S5, B5, mlp_buf_a, MLP_L4, mlp_buf_b, MLP_OUT, false);
  return mlp_buf_b[0];
}

#endif
