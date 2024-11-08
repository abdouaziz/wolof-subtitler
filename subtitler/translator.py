import torch
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import os 
import re 
from google.cloud import translate_v2 as translator


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ="../static/lam_json_data.json"


class Translator:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            "abdouaziiz/m2m100_418M_B30"
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained("abdouaziiz/m2m100_418M_B30")
        self.model.to(self.device)

    def predict(self, text):
        text = str(text).lower()

        model_inputs = self.tokenizer(
            text, max_length=250, truncation=True, return_tensors="pt"
        ).to(self.device)
        gen_tokens = self.model.generate(
            **model_inputs, forced_bos_token_id=self.tokenizer.get_lang_id("wo")
        )
        translate = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        translation = translate[0]
        return translation



class GoogleTranslator:
    def __init__(self, source_language: str = "wo", target_language: str = "fr"):
        self.source_language = source_language
        self.target_language = target_language
        self.translator = translator.Client()

    def translate(self, text: str) -> str:
        self.translated_text = self.translator.translate(
            text.lower(),
            source_language=self.source_language,
            target_language=self.target_language,
        )
        return self.translated_text["translatedText"]

    # call
    def __call__(self, text: str) -> str:
        # remove speaker_1 , 2 , 3
        text = re.sub(r"speaker_\d", "", text)
        translation=self.translate(text)
        return translation



    
    