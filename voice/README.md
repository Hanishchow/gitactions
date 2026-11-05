# Claude voice listener

Always-on mic -> wake word ("Hey Claude ...") -> local Whisper -> Claude Code (bypass permissions) -> spoken reply.

Setup (done once):
1. `claude auth login` (headless Claude needs a valid login)
2. `powershell -File voice\install.ps1` (starts hidden at every logon)
3. `Start-ScheduledTask ClaudeVoice` / `Stop-ScheduledTask ClaudeVoice`

Use: "Hey Claude, open Chrome." Follow-ups within 20 s need no wake word. "Claude stop" aborts.
It pauses itself while another app records (Sound Recorder, calls). Logs: `voice/logs/voice.log` (local, git-ignored).
Settings: `voice/config.toml`. Run in a console for debugging: `voice\.venv\Scripts\python -m voice.listener`.

Safety: the wake word is the only gate unless speaker verification is enabled (`[speaker] verify`), so a video saying
"Hey Claude" could trigger commands. Never done by voice: passwords/payment entry, money transfers, CAPTCHAs, permanent deletion.
