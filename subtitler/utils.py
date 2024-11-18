import librosa
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import librosa
from transformers import Wav2Vec2ForCTC, AutoProcessor
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq




device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


wolof_transcriber = pipeline(
    "automatic-speech-recognition",
    model="derguene/whisper-large-v3-wo", 
    torch_dtype=torch_dtype,
    device=device,)


def transcribe_audio(file_path  ):
    
    audio, sr = librosa.load(file_path)
    
    if sr != 16000:
        audio = librosa.resample(audio , orig_sr = sr, target_sr = sr)
        sr = 16000
        
    transcription = wolof_transcriber(audio,chunk_length_s=30)

    return transcription["text"]   
     

class WhisperTranscriber:
    def __call__(self, path_audio):
        try:
            text = transcribe_audio(path_audio)
            return text
        except Exception as e:
            print(f"Une erreur est survenue lors de la transcription : {e}")
            return None