import configparser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


class LLMService:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.__model_id = config.get("LLM", "MODEL")
        self.__tokenizers = AutoTokenizer.from_pretrained(self.__model_id)
        self.__model = AutoModelForCausalLM.from_pretrained(
            self.__model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def run(self, prompt, max_length):
        messages = [
            {"role": "system", "content": f"Given the following text below, generate a useful response, generate the response in the input language"},
            {"role": "user", "content": prompt},
        ]
        input_ids = self.__tokenizers.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.__model.device)
        streamer = TextStreamer(self.__tokenizers, skip_prompt=True, skip_special_tokens=True)
        generation_output = self.__model.generate(
            input_ids,
            streamer=streamer,
            max_new_tokens=max_length,
            eos_token_id=self.__tokenizers.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )