import threading

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class AlertSystem:
    """Voice alert — new TTS engine per alert avoids Windows threading bugs."""

    def __init__(self):
        self._lock = threading.Lock()
        self.is_playing = False

    def _speak(self, message):
        if pyttsx3 is None:
            print(f"[Alert] pyttsx3 not installed. Message: {message}")
            return
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[Alert Error] {e}")

    def trigger_alert(self, message):
        with self._lock:
            if self.is_playing:
                return False
            self.is_playing = True

        print(f"[ALERT TRIGGERED]: {message}")

        def run():
            try:
                self._speak(message)
            finally:
                with self._lock:
                    self.is_playing = False

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return True
