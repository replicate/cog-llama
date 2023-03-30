from typing import List, Optional
from cog import BasePredictor, Input, ConcatenateIterator
from transformers import LlamaTokenizer
from typing import Any
import torch

from subclass import YieldingLlama

CACHE_DIR = "weights"

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YieldingLlama.from_pretrained("weights/llama-7b", cache_dir=CACHE_DIR, local_files_only=True)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained("weights/tokenizer", cache_dir=CACHE_DIR, local_files_only=True)

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to LLaMA."),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=50
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
            default=1.0
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1
        )) -> ConcatenateIterator[str]:
        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            first_token_yielded = False
            prev_ids = []
            for output in self.model.generate(
                input,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ):
                cur_id = output.item()

                # in order to properly handle spaces, we need to do our own tokenizing. Fun! 
                # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                cur_token = self.tokenizer.convert_ids_to_tokens(cur_id)
                
                # skip initial newline, which this almost always yields. hack - newline id = 13. 
                if not first_token_yielded and not prev_ids and cur_id == 13:
                    continue 
                
                # underscore means a space, means we yield previous tokens
                if cur_token.startswith('▁'): # this is not a standard underscore. 
                    # first token
                    if not prev_ids:
                        prev_ids = [cur_id]
                        continue
                    
                    # there are tokens to yield
                    else:
                        token = self.tokenizer.decode(prev_ids)
                        prev_ids = [cur_id]

                        if not first_token_yielded:
                            # no leading space for first token
                            token = token.strip()
                            first_token_yielded = True
                        yield token
                else:
                    prev_ids.append(cur_id)
                    continue

            token = self.tokenizer.decode(prev_ids)
            yield token
