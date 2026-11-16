"""Decision engine: ripeness index, anomaly flag and firmware status flags -> operator action.

Rules, first match wins:
  1. FLAG_WARMUP                           -> HOLD "warming up"       (severity 0)
  2. FLAG_MQ_FAULT, anomaly, or no index   -> ALERT                   (severity 2)
  3. index >= hi + VENT_MARGIN             -> VENTILATE               (severity 2)
  4. index >= hi                           -> SCRUB                   (severity 1)
  5. otherwise                             -> HOLD, "below target" when index < lo
                                              else "in target window" (severity 0)
Then FLAG_DHT_FAIL lifts severity to at least 1 and appends a note; it never changes the action.

"No index" means None, NaN or infinity. The relay bits (FLAG_VENT, FLAG_SCRUB) only report what the
firmware is already doing, so they are ignored here.
"""
import math
from dataclasses import dataclass, replace
from enum import StrEnum

from ethylene.data import FLAG_DHT_FAIL, FLAG_MQ_FAULT, FLAG_WARMUP

VENT_MARGIN = 0.05


class Action(StrEnum):
    VENTILATE = "VENTILATE"
    SCRUB = "SCRUB"
    HOLD = "HOLD"
    ALERT = "ALERT"


@dataclass(frozen=True)
class Decision:
    """`severity`: 0 ok, 1 watch, 2 act."""

    action: Action
    reason: str
    severity: int

    def to_dict(self) -> dict:
        return {"action": self.action.value, "reason": self.reason, "severity": self.severity}


@dataclass(frozen=True)
class RipeningTarget:
    lo: float = 0.55
    hi: float = 0.85


def decide(
    index: float | None,
    anomaly: bool,
    flags: int,
    target: RipeningTarget = RipeningTarget(),
) -> Decision:
    flags = int(flags)
    idx = float(index) if index is not None and math.isfinite(index) else None
    shown = "" if idx is None else f"{idx:.2f}"

    if flags & FLAG_WARMUP:
        d = Decision(Action.HOLD, "warming up" + (f" (index {shown} ignored)" if shown else ""), 0)
    elif flags & FLAG_MQ_FAULT or anomaly or idx is None:
        causes = []
        if flags & FLAG_MQ_FAULT:
            causes.append("MQ sensor fault")
        if anomaly:
            causes.append("anomalous reading")
        if idx is None:
            causes.append("no valid ripeness index")
        d = Decision(Action.ALERT, ", ".join(causes) + (f" (index {shown})" if shown else ""), 2)
    elif idx >= target.hi + VENT_MARGIN:
        d = Decision(Action.VENTILATE, f"index {shown} is over the ventilate limit {target.hi + VENT_MARGIN:.2f}", 2)
    elif idx >= target.hi:
        d = Decision(Action.SCRUB, f"index {shown} is over the target ceiling {target.hi:.2f}", 1)
    elif idx < target.lo:
        d = Decision(Action.HOLD, f"index {shown} below target (floor {target.lo:.2f})", 0)
    else:
        d = Decision(Action.HOLD, f"index {shown} in target window", 0)

    if flags & FLAG_DHT_FAIL:
        d = replace(
            d,
            severity=max(d.severity, 1),
            reason=d.reason + "; DHT sensor failed, temperature/humidity compensation unreliable",
        )
    return d
