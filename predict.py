from typing import List, Optional
from cog import BasePredictor, Input
from transformers import T5ForConditionalGeneration, T5Tokenizer
from train import MODEL_OUT_PATH, TOKENIZER_NAME
import torch

MODEL_NAME = MODEL_OUT_PATH


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, local_files_only=True
        )
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_NAME)

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to FLAN-T5."),
        n: int = Input(
            description="Number of output sequences to generate", default=1, ge=1, le=5
        ),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=50,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1,
        ),
    ) -> List[str]:
        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        outputs = self.model.generate(
            input,
            num_return_sequences=n,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        print(f"cur memory: {torch.cuda.memory_allocated()}")
        print(f"max allocated: {torch.cuda.max_memory_allocated()}")
        print(f"peak memory: {torch.cuda.max_memory_reserved()}")
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return out


class EightBitPredictor(Predictor):
    """subclass s.t. we can configure whether a model is loaded in 8bit mode from cog.yaml"""

    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME, local_files_only=True, load_in_8bit=True, device_map="auto"
        )
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
