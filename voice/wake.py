"""Pure text logic: wake word, stop phrases. Testable on any OS."""
import re


def _norm(text):
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()


def parse(text, phrases, stop_phrases, active):
    """Return (kind, command).

    kind: 'ignore' | 'stop' | 'wake' (wake word only) | 'command'
    active: True while inside the follow-up window (wake word not required).
    """
    t = _norm(text)
    if not t:
        return "ignore", ""
    words = t.split()
    wake_re = "|".join(re.escape(p) for p in phrases)
    m = re.match(rf"^(?:(?:hey|hi|ok|okay|a)\s+)?(?:{wake_re})\b\s*(.*)$", t)
    heard_wake = bool(m)
    rest = m.group(1).strip() if m else t
    if not heard_wake and not active:
        return "ignore", ""
    if any(rest == s or rest == f"claude {s}" for s in stop_phrases):
        return "stop", ""
    if heard_wake and not rest:
        return "wake", ""
    if len(rest.split()) < 1:
        return "ignore", ""
    return "command", rest
