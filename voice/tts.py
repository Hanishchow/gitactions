"""Offline text-to-speech through Windows SAPI (pyttsx3)."""
import re


def shorten(text, limit=280):
    text = re.sub(r"[`*#_>]+", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end + 1] if end > 60 else cut


def speak(text):
    """Blocking; a fresh engine each time avoids pyttsx3's stuck-loop bug."""
    text = shorten(text)
    if not text:
        return
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", 185)
        eng.say(text)
        eng.runAndWait()
        eng.stop()
    except Exception as e:  # never let TTS kill the listener
        print("[tts error]", e)
