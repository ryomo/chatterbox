import torchaudio as ta
import torch
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

text = "Hello. how are you? This is the Chatterbox multilingual text-to-speech model, it supports 23 languages."
wav = multilingual_model.generate(text, language_id="en")
ta.save("test-en.wav", wav, multilingual_model.sr)

# text = "こんにちは。元気ですか？これは多言語テキスト音声合成モデルのChatterboxです。23の言語に対応しています。"
text = "こんにちわ。元気ですか？これは多言語テキスト音声合成モデルのチャッターボックスです。二十三の言語に対応しています。"  # Generates more natural Japanese pronunciation
wav = multilingual_model.generate(text, language_id="ja")
ta.save("test-ja.wav", wav, multilingual_model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
if Path(AUDIO_PROMPT_PATH).exists():
    wav = multilingual_model.generate(text, language_id="en", audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save("test-voice-clone.wav", wav, multilingual_model.sr)
else:
    print(f"Warning: audio prompt file '{AUDIO_PROMPT_PATH}' not found, skipping voice cloning example.")
