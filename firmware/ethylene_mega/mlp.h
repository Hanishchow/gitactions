// Generic small MLP: int8 weights with per-row float scales, read from PROGMEM. The network shape and all
// tables come from the generated model_weights.h (python -m ethylene.export), so any 6-<hidden>-1 ReLU net
// whose weights fit one AVR object (<= 32767 bytes) runs here without code changes.
//   a0 = (x - SCALER_MEAN) / SCALER_STD
//   a_out[j] = (sum_i W[j*in + i] * a_in[i]) * S[j] + B[j]      ReLU after every layer except the last.
// W_ALL / S_ALL / B_ALL hold the layers one after another; MLP_SIZES = {inputs, layer widths...}.
// Activations live in two MLP_MAX_WIDTH-float ping-pong buffers, nothing else is allocated.
//
// pgm_read_* use 16-bit (near) addresses: fine while the PROGMEM tables stay in the low 64 KB of flash.
#ifndef ETHYLENE_MLP_H
#define ETHYLENE_MLP_H

#include <avr/pgmspace.h>
#include "model_weights.h"

static float mlp_buf_a[MLP_MAX_WIDTH];
static float mlp_buf_b[MLP_MAX_WIDTH];

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
  float *in = mlp_buf_a;
  float *out = mlp_buf_b;
  for (uint8_t i = 0; i < MLP_IN; i++) {
    in[i] = (x[i] - pgm_read_float(SCALER_MEAN + i)) / pgm_read_float(SCALER_STD + i);
  }
  const int8_t *w = W_ALL;
  const float *s = S_ALL;
  const float *b = B_ALL;
  uint16_t n_in = pgm_read_word(MLP_SIZES);
  for (uint8_t l = 0; l < MLP_NLAYERS; l++) {
    uint16_t n_out = pgm_read_word(MLP_SIZES + l + 1);
    mlp_layer(w, s, b, in, n_in, out, n_out, l < MLP_NLAYERS - 1);
    w += (uint16_t)(n_in * n_out);
    s += n_out;
    b += n_out;
    float *tmp = in;
    in = out;
    out = tmp;
    n_in = n_out;
  }
  return in[0];  // the last layer's output is in the buffer that was just swapped into `in`
}

#endif
