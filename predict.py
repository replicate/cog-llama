from typing import List
from cog import BasePredictor, Input
from transformers import T5ForConditionalGeneration, T5Tokenizer

CACHE_DIR = 'weights'
SEP = "<sep>"

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", cache_dir=CACHE_DIR, local_files_only=True)
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", cache_dir=CACHE_DIR, local_files_only=True)

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to FLAN-T5. Multiple prompts supported, split with '{SEP}'"),
        max_length: int = Input(
            description="Maximum number of tokens to generate; a word is generally 2-3 tokens",
            ge=1,
            default=50
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, 1 is random and 0 is deterministic",
            ge=0.01,
            le=1,
            default=0.8,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition",
            ge=1,
            le=10,
            default=1
        ),
        ) -> List[str]:
        prompts = prompt.split(SEP)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(self.device)

        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return out
        