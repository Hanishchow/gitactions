from voice.micwatch import MicWatcher, in_use, normalize


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(entries, **kw):
    clock = Clock()
    box = {"e": entries}
    w = MicWatcher(clock=clock, reader=lambda: box["e"], **kw)
    return w, clock, box


def test_in_use():
    assert in_use(5, 0)
    assert in_use(10, 5)
    assert not in_use(5, 9)
    assert not in_use(0, 0)


def test_normalize():
    assert normalize("C:#Users#me#python.exe") == "c:" + chr(92) + "users" + chr(92) + "me" + chr(92) + "python.exe"


def test_pause_and_resume_with_debounce():
    w, clock, box = make([("zoom.exe", 10, 0)], pause_after=1, resume_after=2)
    assert w.poll() is False          # just started
    clock.t = 1.0
    assert w.poll() is True           # held for 1s -> pause
    assert w.blockers == ["zoom.exe"]
    box["e"] = [("zoom.exe", 10, 20)]
    clock.t = 2.0
    assert w.poll() is True           # free but not for 2s yet
    clock.t = 4.1
    assert w.poll() is False          # resumed


def test_own_and_ignored_apps_do_not_pause():
    w, clock, box = make([], ignore=["alexa"], pause_after=0)
    own = next(iter(w.own)).replace("\\", "#")
    box["e"] = [(own, 10, 0), ("AmazonAlexa_x", 10, 0)]
    clock.t = 5
    assert w.poll() is False


def test_registry_error_means_not_paused():
    def boom():
        raise OSError

    w = MicWatcher(reader=boom)
    assert w.poll() is False
