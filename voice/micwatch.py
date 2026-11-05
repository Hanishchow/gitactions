"""Detect when ANOTHER app is using the microphone (Windows privacy ConsentStore).

An app is capturing when LastUsedTimeStart > 0 and LastUsedTimeStop == 0
(or Start > Stop). winreg is imported lazily so this module imports on any OS.
"""
import sys
import time

CONSENT_KEY = (r"SOFTWARE\Microsoft\Windows\CurrentVersion"
               r"\CapabilityAccessManager\ConsentStore\microphone")


def in_use(start, stop):
    """True while the app has the mic open."""
    return start > 0 and start > stop


def normalize(name):
    """NonPackaged keys encode the exe path with '#' instead of '\'."""
    return name.replace("#", "\\").lower()


def own_executables():
    paths = {sys.executable, getattr(sys, "_base_executable", sys.executable)}
    return {normalize(p) for p in paths if p}


def read_entries():
    """Return [(name, start, stop)] for every app known to the mic ConsentStore."""
    import winreg
    out = []

    def scan(root, prefix=""):
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if sub == "NonPackaged" and not prefix:
                with winreg.OpenKey(root, sub) as np_key:
                    scan(np_key, "np:")
                continue
            try:
                with winreg.OpenKey(root, sub) as k:
                    start = winreg.QueryValueEx(k, "LastUsedTimeStart")[0]
                    stop = winreg.QueryValueEx(k, "LastUsedTimeStop")[0]
            except OSError:
                continue
            out.append((sub, int(start), int(stop)))

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, CONSENT_KEY) as root:
        scan(root)
    return out


class MicWatcher:
    """Debounced pause/resume decision.

    pause_after: seconds another app must hold the mic before we pause.
    resume_after: seconds the mic must be free before we resume.
    ignore: substrings (case-insensitive) of app names that never count
            (apps that hold the mic open while idle).
    """

    def __init__(self, ignore=(), pause_after=1.0, resume_after=2.0,
                 clock=time.monotonic, reader=read_entries):
        self.ignore = [s.lower() for s in ignore]
        self.pause_after = pause_after
        self.resume_after = resume_after
        self.clock = clock
        self.reader = reader
        self.own = own_executables()
        self.paused = False
        self._busy_since = None
        self._free_since = None
        self.blockers = []

    def _counts(self, name):
        n = normalize(name)
        if n in self.own:
            return False
        return not any(s in n for s in self.ignore)

    def poll(self):
        """Read the registry once and return True if we should be paused."""
        try:
            entries = self.reader()
        except OSError:
            entries = []
        self.blockers = [n for n, a, b in entries
                         if in_use(a, b) and self._counts(n)]
        now = self.clock()
        if self.blockers:
            self._free_since = None
            if self._busy_since is None:
                self._busy_since = now
            if not self.paused and now - self._busy_since >= self.pause_after:
                self.paused = True
        else:
            self._busy_since = None
            if self._free_since is None:
                self._free_since = now
            if self.paused and now - self._free_since >= self.resume_after:
                self.paused = False
        return self.paused


if __name__ == "__main__":  # quick manual check: python -m voice.micwatch
    for name, a, b in read_entries():
        print(f"{'IN USE' if in_use(a, b) else 'idle  '}  {name}")
