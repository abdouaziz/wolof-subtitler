from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
 



class WhisperTranscriber:
    def __init__(self, model_name="LiquAId/whisper-tiny-french-HanNeurAI"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
                                                                     
    def transcriber(self ,path_audio):
        audio_input = librosa.load(path_audio, sr=16000)[0]
        input_features = self.processor(
            audio_input, sampling_rate=16000, return_tensors="pt"
        ).input_features

        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription[0]
    
    def __call__(self, path_audio):
        return self.transcriber(path_audio)
    
    
    
    


