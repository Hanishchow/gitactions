"""Always-on voice listener for Claude.  Run:  python -m voice.listener

mic -> energy VAD -> (speaker check) -> Whisper -> wake word -> Claude -> spoken reply
Pauses and releases the mic while any other app is recording.
"""
import collections
import logging
import logging.handlers
import queue
import threading
import time
import tomllib
from pathlib import Path

import numpy as np
import sounddevice as sd

from . import tts, wake
from .agent import Agent
from .micwatch import MicWatcher

HERE = Path(__file__).parent
SR = 16000


def load_config():
    with open(HERE / "config.toml", "rb") as f:
        return tomllib.load(f)


def setup_log():
    (HERE / "logs").mkdir(exist_ok=True)
    log = logging.getLogger("voice")
    log.setLevel(logging.INFO)
    h = logging.handlers.RotatingFileHandler(
        HERE / "logs" / "voice.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(h)
    log.addHandler(logging.StreamHandler())
    return log


def find_device(cfg):
    want, api = cfg["device"].lower(), cfg.get("hostapi", "").lower()
    apis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and want in d["name"].lower() \
                and api in apis[d["hostapi"]]["name"].lower():
            return i
    raise SystemExit(f"No input device matching {cfg['device']!r} ({cfg.get('hostapi')})")


class Listener:
    def __init__(self):
        self.cfg = load_config()
        self.log = setup_log()
        c = self.cfg
        self.frame = int(SR * c["audio"]["frame_ms"] / 1000)
        self.dev = find_device(c["audio"])
        self.q = queue.Queue()
        self.stream = None
        self.speaking = False
        self.busy = False
        self.active_until = 0.0
        self.floor = 0.005
        mw = c["mic_watch"]
        self.watch = MicWatcher(mw["ignore"], mw["pause_after_s"], mw["resume_after_s"])
        self.agent = Agent(c["claude"]["workdir"])
        self.speaker = None
        if c["speaker"]["verify"]:
            from .speaker import SpeakerGate
            self.speaker = SpeakerGate(c["speaker"]["threshold"])
        from faster_whisper import WhisperModel
        self.log.info("loading Whisper %s ...", c["stt"]["model"])
        self.stt = WhisperModel(c["stt"]["model"], device="cpu", compute_type=c["stt"]["compute_type"])

    # ---- audio plumbing -------------------------------------------------
    def _cb(self, indata, frames, t, status):
        if not self.speaking:
            self.q.put(indata[:, 0].copy())

    def open_mic(self):
        if self.stream is None:
            self.stream = sd.InputStream(device=self.dev, samplerate=SR, channels=1,
                                         dtype="float32", blocksize=self.frame, callback=self._cb)
            self.stream.start()
            self.log.info("mic open")

    def close_mic(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self._flush()

    def _flush(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def _watch_loop(self):
        while True:
            try:
                self.watch.poll()
            except Exception as e:
                self.log.info("watch error %s", e)
            time.sleep(self.cfg["mic_watch"]["poll_s"])

    # ---- speech ---------------------------------------------------------
    def say(self, text):
        self.speaking = True
        try:
            tts.speak(text)
        finally:
            time.sleep(0.35)
            self._flush()
            self.speaking = False

    def transcribe(self, audio):
        segs, _ = self.stt.transcribe(
            audio, language="en", beam_size=1, condition_on_previous_text=False,
            initial_prompt="Hey Claude, open Chrome. Claude, stop.")
        return " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6).strip()

    def run_command(self, text):
        self.busy = True
        try:
            self.log.info("CMD: %s", text)
            reply = self.agent.ask(text)
            self.log.info("REPLY: %s", reply)
            self.say(reply)
        finally:
            self.busy = False
            self.active_until = time.time() + self.cfg["wake"]["followup_s"]

    def handle(self, audio):
        if self.speaker and not self.speaker.is_owner(audio):
            self.log.info("ignored: voice not recognised")
            return
        text = self.transcribe(audio)
        active = self.busy or time.time() < self.active_until
        kind, cmd = wake.parse(text, self.cfg["wake"]["phrases"],
                               self.cfg["wake"]["stop_phrases"], active)
        if kind == "ignore":
            return
        self.log.info("heard (%s): %s", kind, text)
        if kind == "stop":
            self.agent.stop()
            self.active_until = 0
            threading.Thread(target=self.say, args=("Stopped.",), daemon=True).start()
        elif kind == "wake":
            self.active_until = time.time() + self.cfg["wake"]["followup_s"]
            threading.Thread(target=self.say, args=("Yes?",), daemon=True).start()
        elif kind == "command" and not self.busy:
            threading.Thread(target=self.run_command, args=(cmd,), daemon=True).start()

    # ---- main loop ------------------------------------------------------
    def run(self):
        v = self.cfg["vad"]
        threading.Thread(target=self._watch_loop, daemon=True).start()
        pre = collections.deque(maxlen=int(0.3 * SR / self.frame))
        buf, in_speech, silent, last_state = [], False, 0, None
        end_frames = int(v["end_silence_s"] * SR / self.frame)
        self.log.info("ready. Say 'Hey Claude, ...'")
        while True:
            state = "paused" if self.watch.paused else "listening"
            if state != last_state:
                self.log.info("state: %s%s", state,
                              f" (mic used by {self.watch.blockers})" if state == "paused" else "")
                last_state = state
            if self.watch.paused:
                self.close_mic()
                buf, in_speech = [], False
                time.sleep(0.3)
                continue
            self.open_mic()
            try:
                x = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            rms = float(np.sqrt(np.mean(x * x)))
            thr = max(v["min_rms"], self.floor * v["noise_mult"])
            loud = rms > thr
            if not in_speech:
                self.floor = 0.98 * self.floor + 0.02 * rms
                pre.append(x)
                if loud:
                    in_speech, buf, silent = True, list(pre), 0
            else:
                buf.append(x)
                silent = 0 if loud else silent + 1
                dur = len(buf) * self.frame / SR
                if silent >= end_frames or dur >= v["max_utterance_s"]:
                    in_speech = False
                    speech_s = (len(buf) - silent) * self.frame / SR
                    audio = np.concatenate(buf)
                    buf = []
                    if speech_s >= v["min_speech_s"]:
                        try:
                            self.handle(audio)
                        except Exception as e:
                            self.log.info("error: %r", e)


if __name__ == "__main__":
    try:
        Listener().run()
    except KeyboardInterrupt:
        pass
