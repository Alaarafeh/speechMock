import importlib.util

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

has_cuda = torch.cuda.is_available()
has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
has_flash_attn = importlib.util.find_spec("flash_attn") is not None

if has_cuda:
    model_kwargs = {"device_map": "cuda:0", "dtype": torch.bfloat16}
    model_kwargs["attn_implementation"] = "flash_attention_2" if has_flash_attn else "sdpa"
elif has_mps:
    model_kwargs = {"device_map": "mps", "dtype": torch.float16}
else:
    model_kwargs = {"device_map": "cpu", "dtype": torch.float32}

# Load the model
try:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        **model_kwargs,
    )
except Exception as exc:
    if has_mps:
        print(f"MPS load failed ({exc}); falling back to CPU.")
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cpu",
            dtype=torch.float32,
        )
    else:
        raise

# Custom Voice Generation
wavs, sr = model.generate_custom_voice(
    text="I am robot",
    language="English",
    speaker="Vivian",
    instruct="say it in a very happy tone",
)
sf.write("output.wav", wavs[0], sr)
