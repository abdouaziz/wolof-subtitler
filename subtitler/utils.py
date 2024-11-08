import librosa
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import librosa
from transformers import Wav2Vec2ForCTC, AutoProcessor



model_id = "facebook/mms-1b-all"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)


processor.tokenizer.set_target_lang("wol")
model.load_adapter("wol")


wolof_transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs={"attn_implementation": "flash_attention_2"}
    if is_flash_attn_2_available()
    else {"attn_implementation": "sdpa"},
)


def transcribe_audio(path_audio):

    audio, sr = librosa.load(path_audio)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    transcription = wolof_transcriber(audio, chunk_length_s=30, batch_size=24)

    return transcription["text"]


class WhisperTranscriber:
    def __call__(self, path_audio):
        return transcribe_audio(path_audio)
