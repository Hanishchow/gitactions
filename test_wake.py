from voice.wake import parse

P = ["claude", "cloud", "clod"]
S = ["stop", "cancel"]


def test_ignored_without_wake_word():
    assert parse("open chrome please", P, S, False) == ("ignore", "")


def test_wake_with_command():
    assert parse("Hey Claude, open Chrome.", P, S, False) == ("command", "open chrome")
    assert parse("hey cloud open notepad", P, S, False) == ("command", "open notepad")


def test_wake_only():
    assert parse("Hey Claude.", P, S, False) == ("wake", "")


def test_followup_needs_no_wake_word():
    assert parse("now open youtube", P, S, True) == ("command", "now open youtube")


def test_stop():
    assert parse("Claude, stop", P, S, False) == ("stop", "")
    assert parse("stop", P, S, True) == ("stop", "")
    assert parse("stop", P, S, False) == ("ignore", "")


def test_empty():
    assert parse("", P, S, True) == ("ignore", "")
