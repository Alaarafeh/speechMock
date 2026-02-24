import shutil
import subprocess

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

REF_AUDIO_PATH = "my_voice.wav"
REF_AUDIO_SECONDS = 8
REF_TEXT = "hello this is my voice sample"


def record_reference_audio(path: str, seconds: int) -> None:
    if shutil.which("sox") is None:
        raise RuntimeError("SoX is not installed. Install it with: brew install sox")

    print(f"Recording starts now for {seconds} seconds. Speak into your mic.")
    subprocess.run(["sox", "-d", path, "trim", "0", str(seconds)], check=True)
    print(f"Saved reference audio to {path}")


has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

record_reference_audio(REF_AUDIO_PATH, REF_AUDIO_SECONDS)

try:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="mps" if has_mps else "cpu",
        dtype=torch.float16 if has_mps else torch.float32,
    )
except Exception as exc:
    if has_mps:
        print(f"MPS load failed ({exc}); falling back to CPU.")
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cpu",
            dtype=torch.float32,
        )
    else:
        raise

wavs, sr = model.generate_voice_clone(
    text="Hello, this is my cloned voice.",
    language="English",
    ref_audio=REF_AUDIO_PATH,
    ref_text=REF_TEXT,  # Keep this exactly what you say while recording.
    x_vector_only_mode=False,                 # better quality if ref_text is correct
)

sf.write("output.wav", wavs[0], sr)
