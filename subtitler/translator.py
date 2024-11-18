import os 
import re 
from google.cloud import translate_v2 as translator


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ="/Users/modoudiakhate/Documents/Projets/LAM SUBTITLER/static/lam_json_data.json"


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
    
    
if __name__=="__main__":
    translator = GoogleTranslator()
    print(translator("Nopp naa ko ci xam-xam bi."))
 