from typing import List, Optional
from collections import OrderedDict
from cog import BasePredictor, Input, Path
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoConfig, AutoModelForSeq2SeqLM
from train import resolve_model, load_tokenizer, MODEL_NAME, DEFAULT_MODEL_NAME
import os
import torch
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
import time

#os.environ['COG_WEIGHTS'] = 'https://pbxt.replicate.delivery/3zc9rpb6wG66M9lwNCLbL4V1Lywjfg2Zi5eco8CMA84B0LtQA/tuned_weights.tensors'

# if 'COG_WEIGHTS' not in os.environ:
#     os.environ['COG_WEIGHTS'] = DEFAULT_MODEL_NAME

class Predictor(BasePredictor):
    def setup(self, weights:Optional[Path] = None):
        if weights is not None and weights.name == 'weights':
            weights = None
        print(weights)
        # returns "weights"
        # FIXME - we should use the actual "weights" argument here unless we decide not to
        weights = None if 'COG_WEIGHTS' not in os.environ else os.environ["COG_WEIGHTS"]
        print('weights path', weights)
        model_name = resolve_model(weights)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if 'tensors' in model_name: #TODO: this is not the best way to determine whether something is or is not tensorized.
            self.model = self.load_tensorizer(model_name)
        else:
            st = time.time()
            print(f'loading weights from {model_name} w/o tensorizer')
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, cache_dir='pretrained_weights', torch_dtype=torch.float16, local_files_only=True
            )
            self.model.to(self.device)
            print(f'weights loaded in {time.time() - st}')
        self.tokenizer = load_tokenizer()

    def load_tensorizer(self, weights):
        st = time.time()
        print(f'deserializing weights from {weights}')
        config = AutoConfig.from_pretrained(MODEL_NAME)

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
        model_name = resolve_model(weights)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, local_files_only=True, load_in_8bit=True, device_map="auto"
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
