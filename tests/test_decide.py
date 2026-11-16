import dataclasses
import json
import math

import numpy as np
import pytest

from ethylene.data import FLAG_DHT_FAIL, FLAG_MQ_FAULT, FLAG_SCRUB, FLAG_VENT, FLAG_WARMUP
from ethylene.decide import Action, Decision, RipeningTarget, decide

NAN = float("nan")
INDICES = [None, NAN, 0.30, 0.55, 0.70, 0.85, 0.86, 0.91, 1.0]


# --- types -----------------------------------------------------------------


def test_action_values_are_the_wire_strings():
    assert [a.value for a in Action] == ["VENTILATE", "SCRUB", "HOLD", "ALERT"]
    assert Action.HOLD == "HOLD"
    assert isinstance(Action.ALERT, str)


def test_default_target():
    t = RipeningTarget()
    assert (t.lo, t.hi) == (0.55, 0.85)


def test_decision_is_frozen_and_serialises():
    d = Decision(Action.SCRUB, "why", 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.severity = 2
    out = d.to_dict()
    assert out == {"action": "SCRUB", "reason": "why", "severity": 1}
    assert type(out["action"]) is str
    assert json.loads(json.dumps(out)) == out


# --- warm-up beats everything ----------------------------------------------


@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("anomaly", [False, True])
@pytest.mark.parametrize("extra", [0, FLAG_MQ_FAULT, FLAG_VENT, FLAG_SCRUB, FLAG_MQ_FAULT | FLAG_VENT | FLAG_SCRUB])
def test_warmup_suppresses_every_action(index, anomaly, extra):
    d = decide(index, anomaly, FLAG_WARMUP | extra)
    assert d.action is Action.HOLD
    assert "warming up" in d.reason
    assert d.severity == 0


def test_warmup_reason_mentions_index_when_present():
    assert "0.95" in decide(0.95, False, FLAG_WARMUP).reason
    assert "0." not in decide(None, False, FLAG_WARMUP).reason


# --- faults never read as fine ---------------------------------------------


@pytest.mark.parametrize("index", INDICES)
def test_mq_fault_is_alert_whatever_the_index(index):
    d = decide(index, False, FLAG_MQ_FAULT)
    assert d.action is Action.ALERT
    assert d.severity == 2
    assert "MQ" in d.reason


@pytest.mark.parametrize("index", INDICES)
def test_anomaly_is_alert_whatever_the_index(index):
    d = decide(index, True, 0)
    assert d.action is Action.ALERT
    assert d.severity == 2
    assert "anomal" in d.reason


@pytest.mark.parametrize("index", [None, NAN, np.float32("nan"), np.float64("nan"), math.inf, -math.inf])
def test_missing_or_non_finite_index_is_alert(index):
    d = decide(index, False, 0)
    assert d.action is Action.ALERT
    assert d.severity == 2
    assert "no valid" in d.reason


def test_mq_fault_and_anomaly_both_reported():
    d = decide(0.70, True, FLAG_MQ_FAULT)
    assert d.action is Action.ALERT
    assert "MQ" in d.reason and "anomal" in d.reason


# --- the ripening bands ----------------------------------------------------


@pytest.mark.parametrize(
    "index, action, severity, phrase",
    [
        (0.30, Action.HOLD, 0, "below target"),
        (0.0, Action.HOLD, 0, "below target"),
        (0.5499, Action.HOLD, 0, "below target"),
        (0.55, Action.HOLD, 0, "in target window"),  # lo is inside the window
        (0.70, Action.HOLD, 0, "in target window"),
        (0.8499, Action.HOLD, 0, "in target window"),
        (0.85, Action.SCRUB, 1, ""),  # hi is the first SCRUB value
        (0.86, Action.SCRUB, 1, ""),
        (0.8999, Action.SCRUB, 1, ""),
        (0.90, Action.VENTILATE, 2, ""),  # hi + 0.05 is the first VENTILATE value
        (0.91, Action.VENTILATE, 2, ""),
        (1.0, Action.VENTILATE, 2, ""),
    ],
)
def test_bands_and_boundaries(index, action, severity, phrase):
    d = decide(index, False, 0)
    assert (d.action, d.severity) == (action, severity)
    assert phrase in d.reason


def test_below_target_does_not_claim_the_window():
    assert "in target window" not in decide(0.30, False, 0).reason
    assert "below target" not in decide(0.70, False, 0).reason


@pytest.mark.parametrize("index, text", [(0.7, "0.70"), (0.30, "0.30"), (0.8567, "0.86"), (0.91, "0.91"), (0.5, "0.50")])
def test_reason_mentions_index_to_two_decimals(index, text):
    assert text in decide(index, False, 0).reason


def test_alert_reasons_mention_index_only_when_present():
    assert "0.70" in decide(0.70, True, 0).reason
    assert "0.70" in decide(0.70, False, FLAG_MQ_FAULT).reason
    assert "0." not in decide(None, True, FLAG_MQ_FAULT).reason


# --- relay bits and unknown bits do not influence the decision --------------


@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("anomaly", [False, True])
@pytest.mark.parametrize("flags", [FLAG_VENT, FLAG_SCRUB, FLAG_VENT | FLAG_SCRUB, 0x100, 0xFF00])
def test_relay_and_unknown_bits_are_ignored(index, anomaly, flags):
    assert decide(index, anomaly, flags) == decide(index, anomaly, 0)


# --- DHT failure ------------------------------------------------------------


def test_dht_fail_in_window_holds_at_severity_one():
    d = decide(0.70, False, FLAG_DHT_FAIL)
    assert d.action is Action.HOLD
    assert d.severity == 1
    assert "in target window" in d.reason
    assert "DHT" in d.reason


@pytest.mark.parametrize(
    "index, anomaly, flags, action, severity",
    [
        (0.30, False, FLAG_DHT_FAIL, Action.HOLD, 1),
        (0.86, False, FLAG_DHT_FAIL, Action.SCRUB, 1),
        (0.91, False, FLAG_DHT_FAIL, Action.VENTILATE, 2),  # never lowered
        (None, False, FLAG_DHT_FAIL, Action.ALERT, 2),
        (0.70, True, FLAG_DHT_FAIL, Action.ALERT, 2),
        (0.70, False, FLAG_DHT_FAIL | FLAG_MQ_FAULT, Action.ALERT, 2),
        (0.91, False, FLAG_DHT_FAIL | FLAG_WARMUP, Action.HOLD, 1),
    ],
)
def test_dht_fail_only_raises_severity(index, anomaly, flags, action, severity):
    d = decide(index, anomaly, flags)
    assert (d.action, d.severity) == (action, severity)
    assert "DHT" in d.reason


def test_dht_fail_never_changes_the_action():
    for index in INDICES:
        for anomaly in (False, True):
            for base in (0, FLAG_MQ_FAULT, FLAG_WARMUP):
                assert decide(index, anomaly, base | FLAG_DHT_FAIL).action is decide(index, anomaly, base).action


def test_no_dht_note_without_the_flag():
    assert "DHT" not in decide(0.70, False, 0).reason


# --- custom target ----------------------------------------------------------


def test_custom_target_moves_the_bands():
    t = RipeningTarget(lo=0.20, hi=0.50)
    assert decide(0.10, False, 0, t).action is Action.HOLD
    assert "below target" in decide(0.10, False, 0, t).reason
    assert "in target window" in decide(0.30, False, 0, t).reason
    assert decide(0.50, False, 0, t).action is Action.SCRUB
    assert decide(0.54, False, 0, t).action is Action.SCRUB
    assert decide(0.55, False, 0, t).action is Action.VENTILATE


def test_default_target_is_not_shared_mutable_state():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RipeningTarget().hi = 0.5


# --- numpy inputs (the model and CSV layers hand these in) -------------------


def test_numpy_scalars_are_accepted():
    d = decide(np.float32(0.70), np.bool_(False), np.int32(FLAG_DHT_FAIL))
    assert (d.action, d.severity) == (Action.HOLD, 1)
    assert decide(np.float32(0.91), np.bool_(False), np.int32(0)).action is Action.VENTILATE
    assert decide(np.float32(0.70), np.bool_(True), np.int32(0)).action is Action.ALERT
    assert decide(np.float32(0.70), np.bool_(False), np.int32(FLAG_MQ_FAULT)).action is Action.ALERT
    assert decide(np.float32(0.70), np.bool_(False), np.int32(FLAG_WARMUP)).action is Action.HOLD
