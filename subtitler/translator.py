import torch
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration


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
