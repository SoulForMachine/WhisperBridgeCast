from app.server.transl_backends.base import TranslBase


class EuroLLM(TranslBase):
    @classmethod
    def get_name(cls) -> str:
        return "EuroLLM"

    def __init__(self, transl_params: dict, src_lang: str, target_lang: str):
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        transl_params = transl_params or {}
        api_key = transl_params.get("api_key", "")
        login(token=api_key)

        model_id = "utter-project/EuroLLM-1.7B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
        import torch

        self.model = torch.compile(self.model)

        self.src_lang = src_lang
        self.target_lang = target_lang

    def translate_text(self, text: str) -> str:
        if not text.endswith((".", "!", "?", ";")):
            text += "..."
        prompt = f"{self.src_lang}: {text} {self.target_lang}:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        transl_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        transl_text = transl_text[len(prompt) :].strip()
        return transl_text
