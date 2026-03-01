from __future__ import annotations


class VoiceInput:
    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self.sample_rate = 16000
        self._model = None
        self._sounddevice = None

        try:
            import whisper  # type: ignore
            import sounddevice as sd  # type: ignore

            self._model = whisper.load_model(model_name)
            self._sounddevice = sd
        except Exception:
            self._model = None
            self._sounddevice = None

    def record_and_transcribe(self, duration: float = 5.0) -> str:
        if self._model is None or self._sounddevice is None:
            return input("[Voice unavailable, type command] > ").strip()

        print("[Recording...]")
        audio = self._sounddevice.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        self._sounddevice.wait()
        print("[Transcribing...]")
        result = self._model.transcribe(audio.flatten())
        return str(result.get("text", "")).strip()
