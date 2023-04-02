import time
from collections import OrderedDict
from typing import List, Optional

import torch
from cog import BasePredictor, Input, Path
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import (AutoConfig, AutoModelForSeq2SeqLM,
                          T5ForConditionalGeneration, T5Tokenizer)

from config import HUGGINGFACE_MODEL_NAME, load_tokenizer


class Predictor(BasePredictor):
    def setup(self, weights:Optional[Path] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if weights is not None and weights.name == 'weights':
            # bugfix
            weights = None

        if weights is None:
            self.model = self.load_huggingface_model(weights=HUGGINGFACE_MODEL_NAME)
        elif 'tensors' in weights.filename: # assuming URLPath for now
            self.model = self.load_tensorizer(weights)
        else:
            self.model = self.load_huggingface_model(weights=weights)

        self.tokenizer = load_tokenizer()

    def load_huggingface_model(self, weights=None):
        st = time.time()
        print(f'loading weights from {weights} w/o tensorizer')
        model = T5ForConditionalGeneration.from_pretrained(
            weights, cache_dir='pretrained_weights', torch_dtype=torch.float16
        )
        model.to(self.device)
        print(f'weights loaded in {time.time() - st}')
        return model


    def load_tensorizer(self, weights):
        st = time.time()
        print(f'deserializing weights from {weights}')
        config = AutoConfig.from_pretrained(HUGGINGFACE_MODEL_NAME)

        model = no_init_or_tensor(
            lambda: AutoModelForSeq2SeqLM.from_pretrained(
                None, config=config, state_dict=OrderedDict()
            )
        )
        des = TensorDeserializer(weights, plaid_mode=True)
        des.load_into_module(model)
        print(f'weights loaded in {time.time() - st}')
        return model

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
        debug : bool = Input(
            description="provide debugging output in logs",
            default=False
        )
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
        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return out


class EightBitPredictor(Predictor):
    """subclass s.t. we can configure whether a model is loaded in 8bit mode from cog.yaml"""

    def setup(self, weights=None):
        if weights is not None and weights.name == 'weights':
            # bugfix
            weights = None
        # TODO: fine-tuned 8bit weights.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(
            HUGGINGFACE_MODEL_NAME, load_in_8bit=True, device_map="auto"
        )
        self.tokenizer = load_tokenizer()
