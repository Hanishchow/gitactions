// Ethylene instrument firmware, Arduino Mega 2560.
//
// Every SAMPLE_MS: read MQ-135 (A0), LDR (A1), DHT11 (D22); apply the paper's formulas
// (ethylene_math.h); once warm, run the int8 MLP (mlp.h) and drive the vent / scrub relays with
// hysteresis; print one CSV line on Serial0 (columns = CSV_HEADER, printed once at start).
//
// States: WARMUP (power-up heater warm-up, WARMUP_MS; the last BASELINE_MS of it is the baseline
// window: R0 = mean Rs, LDR_blk = mean LDR, so power up in clean air) -> RUN. Sending the byte 'r'
// starts a fresh BASELINE_MS baseline window (heater already warm). Both relays stay off while warming,
// on an MQ fault, without a good DHT reading, or without a finite index.
//
// No String, no dynamic allocation; SRAM use is dominated by the MLP's two 256-float buffers.
#include "config.h"
#include "ethylene_math.h"
#include "dht11.h"
#include "mlp.h"

#define CSV_HEADER "t_ms,adc_mq,rs_r0,c_comp,ldr_raw,ldr_blk,temp_c,rh,idx,flags"

// Baseline needs at least half of its window's samples with a valid Rs, else it starts over.
#define BASELINE_MIN_SAMPLES ((BASELINE_MS / SAMPLE_MS + 1) / 2)

enum Phase : uint8_t { PHASE_WARMUP, PHASE_RUN };

static Phase phase = PHASE_WARMUP;
static unsigned long phase_start_ms = 0;  // start of the current warm-up / re-baseline phase
static unsigned long phase_len_ms = WARMUP_MS;
static unsigned long last_sample_ms = 0;

static float r0_ohm = NAN;
static float ldr_blk = NAN;
static float base_rs_sum = 0.0f, base_ldr_sum = 0.0f;
static uint16_t base_rs_n = 0, base_ldr_n = 0;

static bool have_dht = false;
static int8_t dht_t = 0;
static uint8_t dht_rh = 0;
static uint8_t dht_fails = 0;  // consecutive failed DHT reads

static bool vent_on = false, scrub_on = false;
static bool led_on = false;

// ---------------------------------------------------------------------------------------------
// Drives a relay pin only when RELAYS_ENABLED; otherwise the pins are never touched and the flags
// (FLAG_VENT / FLAG_SCRUB) still report what the relays would be commanded to do.
static void set_relay(uint8_t pin, bool on) {
#if RELAYS_ENABLED
  digitalWrite(pin, (on == (RELAY_ACTIVE_HIGH != 0)) ? HIGH : LOW);
#else
  (void)pin;
  (void)on;
#endif
}

static void relays_off() {
  vent_on = false;
  scrub_on = false;
  set_relay(PIN_VENT, false);
  set_relay(PIN_SCRUB, false);
}

static void start_phase(unsigned long now, unsigned long len) {
  phase = PHASE_WARMUP;
  phase_start_ms = now;
  phase_len_ms = len;
  base_rs_sum = base_ldr_sum = 0.0f;
  base_rs_n = base_ldr_n = 0;
  relays_off();
}

// Mean of ADC_AVG reads about 1 ms apart. `rail` is set if any single read hit 0 or ADC_MAX.
static float read_adc_avg(uint8_t pin, bool *rail) {
  uint16_t sum = 0;
  *rail = false;
  (void)analogRead(pin);  // discard: the first read after a channel switch carries the previous channel
  delay(1);
  for (uint8_t i = 0; i < ADC_AVG; i++) {
    if (i) delay(1);
    uint16_t v = analogRead(pin);
    if (v == 0 || v >= ADC_MAX) *rail = true;
    sum += v;
  }
  return (float)sum / ADC_AVG;
}

// 4 decimals; non-finite -> "nan"; magnitudes the core's printFloat would show as "ovf" -> scientific.
static void print_float(float x) {
  if (isnan(x) || isinf(x)) {
    Serial.print(F("nan"));
  } else if (fabs(x) >= 1.0e9f) {
    char buf[16];
    dtostre(x, buf, 4, 0);
    Serial.print(buf);
  } else {
    Serial.print(x, 4);
  }
}

static void apply_relays(float idx) {
  if (idx >= (float)SCRUB_IDX) scrub_on = true;
  else if (idx < (float)(SCRUB_IDX - RELAY_HYST)) scrub_on = false;
  if (idx >= (float)VENT_IDX) vent_on = true;
  else if (idx < (float)(VENT_IDX - RELAY_HYST)) vent_on = false;
  set_relay(PIN_SCRUB, scrub_on);
  set_relay(PIN_VENT, vent_on);
}

// ---------------------------------------------------------------------------------------------
static void print_csv(unsigned long now, float adc_mq, float rs_r0, float c_comp, float ldr_raw, float ldr_blk_out,
                      float temp_c, float rh, float idx, bool idx_ok, uint8_t flags) {
  Serial.print(now);
  Serial.print(',');
  Serial.print((unsigned int)(adc_mq + 0.5f));
  Serial.print(',');
  print_float(rs_r0);
  Serial.print(',');
  print_float(c_comp);
  Serial.print(',');
  print_float(ldr_raw);
  Serial.print(',');
  print_float(ldr_blk_out);
  Serial.print(',');
  print_float(temp_c);
  Serial.print(',');
  print_float(rh);
  Serial.print(',');
  if (idx_ok) print_float(idx);
  else Serial.print(F("-1"));
  Serial.print(',');
  Serial.println(flags);
}

#if SELFTEST_CMD
// `T,adc_mq,ldr_raw,temp_c,rh,r0_ohm,ldr_blk` -> one CSV line computed by the real chain, flags = FLAG_SELFTEST.
static void run_selftest(const char *args) {
  float v[6];
  const char *p = args;
  for (uint8_t k = 0; k < 6; k++) {
    char *end;
    v[k] = (float)strtod(p, &end);
    if (end == p) {
      Serial.println(F("# selftest: expected T,adc_mq,ldr_raw,temp_c,rh,r0_ohm,ldr_blk"));
      return;
    }
    p = (*end == ',') ? end + 1 : end;
  }
  float adc_mq = v[0], ldr_raw = v[1], temp_c = v[2], rh = v[3], r0 = v[4], ldr_b = v[5];
  float rs = adc_to_rs(adc_mq);
  float rs_r0 = rs / r0;
  float c_comp = compensate(concentration(rs_r0), temp_c, rh);
  float idx = NAN;
  if (isfinite(rs_r0) && isfinite(c_comp)) {
    float x[MLP_IN] = {rs_r0, c_comp, ldr_raw, ldr_b, temp_c, rh};
    idx = mlp_forward(x);
  }
  print_csv(millis(), adc_mq, rs_r0, c_comp, ldr_raw, ldr_b, temp_c, rh, idx, isfinite(idx), FLAG_SELFTEST);
}
#endif

static char cmd_buf[80];
static uint8_t cmd_len = 0;

static void handle_command(const char *cmd) {
#if SELFTEST_CMD
  if (cmd[0] == 'T' && cmd[1] == ',') run_selftest(cmd + 2);
#else
  (void)cmd;
#endif
}

static void sample(unsigned long now) {
  uint8_t flags = 0;

  // Close the warm-up / baseline phase when its time is up.
  if (phase == PHASE_WARMUP && now - phase_start_ms >= phase_len_ms) {
    if (base_rs_n >= BASELINE_MIN_SAMPLES && base_ldr_n > 0) {
      r0_ohm = base_rs_sum / base_rs_n;
      ldr_blk = base_ldr_sum / base_ldr_n;
      phase = PHASE_RUN;
    } else {
      start_phase(now, BASELINE_MS);  // too many MQ faults in the window: baseline again
    }
  }

  bool mq_rail, ldr_rail;
  float adc_mq = read_adc_avg(PIN_MQ, &mq_rail);
  float ldr_raw = read_adc_avg(PIN_LDR, &ldr_rail);  // LDR rails are legal (dark / bright)
  float rs = mq_rail ? NAN : adc_to_rs(adc_mq);
  if (isnan(rs)) flags |= FLAG_MQ_FAULT;

  int8_t t;
  uint8_t h;
  if (dht11_read(PIN_DHT, &t, &h) && dht_in_range((float)t, (float)h)) {
    dht_t = t;
    dht_rh = h;
    have_dht = true;
    dht_fails = 0;
  } else {
    flags |= FLAG_DHT_FAIL;  // reuse the last good values for up to DHT_MAX_FAILS samples
    if (dht_fails < 255) dht_fails++;
    if (dht_fails >= DHT_MAX_FAILS) have_dht = false;  // too stale to trust: no index until it recovers
  }
  float temp_c = have_dht ? (float)dht_t : NAN;
  float rh = have_dht ? (float)dht_rh : NAN;

  float rs_r0 = NAN, c_comp = NAN, ldr_blk_out = NAN;
  float idx = NAN;

  if (phase == PHASE_WARMUP) {
    flags |= FLAG_WARMUP;
    unsigned long elapsed = now - phase_start_ms;
    if (elapsed + BASELINE_MS >= phase_len_ms) {  // inside the baseline window
      if (!isnan(rs)) {
        base_rs_sum += rs;
        base_rs_n++;
      }
      base_ldr_sum += ldr_raw;
      base_ldr_n++;
    }
  } else {
    rs_r0 = rs / r0_ohm;  // NaN propagates on an MQ fault
    c_comp = compensate(concentration(rs_r0), temp_c, rh);
    ldr_blk_out = ldr_blk;
    if (!(flags & FLAG_MQ_FAULT) && have_dht && isfinite(rs_r0) && isfinite(c_comp)) {
      float x[MLP_IN] = {rs_r0, c_comp, ldr_raw, ldr_blk, temp_c, rh};
      idx = mlp_forward(x);
    }
  }

  bool idx_ok = isfinite(idx);
  if (idx_ok) {
    apply_relays(idx);
  } else {
    relays_off();
  }
  if (vent_on) flags |= FLAG_VENT;
  if (scrub_on) flags |= FLAG_SCRUB;

  // Status LED: steady on while warming, blinking on an MQ fault, off when running normally.
  if (flags & FLAG_MQ_FAULT) led_on = !led_on;
  else led_on = (phase == PHASE_WARMUP);
  digitalWrite(PIN_LED, led_on ? HIGH : LOW);

  print_csv(now, adc_mq, rs_r0, c_comp, ldr_raw, ldr_blk_out, temp_c, rh, idx, idx_ok, flags);
}

// ---------------------------------------------------------------------------------------------
void setup() {
#if RELAYS_ENABLED
  // Relays: latch the OFF level before switching to output so they never click on at boot.
  set_relay(PIN_VENT, false);
  set_relay(PIN_SCRUB, false);
  pinMode(PIN_VENT, OUTPUT);
  pinMode(PIN_SCRUB, OUTPUT);
  set_relay(PIN_VENT, false);
  set_relay(PIN_SCRUB, false);
#endif
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_DHT, INPUT_PULLUP);

  Serial.begin(SERIAL_BAUD);
  Serial.println(F(CSV_HEADER));

  {  // one-off timing of the MLP on this board; a comment line, ignored by the app's parser
    float probe[MLP_IN] = {1.0f, 1.0f, 600.0f, 600.0f, 25.0f, 60.0f};
    unsigned long t0 = micros();
    volatile float sink = mlp_forward(probe);
    (void)sink;
    Serial.print(F("# mlp_us="));
    Serial.println(micros() - t0);
  }

  unsigned long now = millis();
  start_phase(now, WARMUP_MS);
  last_sample_ms = now - SAMPLE_MS;  // first sample right away
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == 'r' && cmd_len == 0) {
      // 'r' re-baselines, but is ignored during the initial heater warm-up (it would cut it short).
      if (!(phase == PHASE_WARMUP && phase_len_ms == WARMUP_MS)) start_phase(millis(), BASELINE_MS);
    } else if (c == '\n' || c == '\r') {
      if (cmd_len > 0 && cmd_len < sizeof(cmd_buf)) {
        cmd_buf[cmd_len] = 0;
        handle_command(cmd_buf);
      }
      cmd_len = 0;
    } else if (cmd_len < sizeof(cmd_buf) - 1) {
      cmd_buf[cmd_len++] = c;
    } else {
      cmd_len = sizeof(cmd_buf);  // overlong line: ignored until its newline
    }
  }

  unsigned long now = millis();
  if (now - last_sample_ms >= SAMPLE_MS) {
    // Stay on the SAMPLE_MS grid; if a sample overran a whole period, resync instead of bursting.
    last_sample_ms = (now - last_sample_ms >= 2 * SAMPLE_MS) ? now : last_sample_ms + SAMPLE_MS;
    sample(now);
  }
}
