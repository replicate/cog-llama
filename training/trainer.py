import argparse
import copy
import json
import os
import time
import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
from cog import Input, Path
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from torch.utils.data import Dataset
from transformers import LlamaForCausalLM, Trainer, TrainingArguments,AutoConfig, DataCollatorForSeq2Seq
import transformers

from config import DEFAULT_MODEL_NAME, load_tokenizer, CONFIG_LOCATION

MODEL_OUT = "/src/tuned_weights.tensors"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
DIST_OUT_DIR = "tmp/model"
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

class DatasetBuilder:
    """Dataset agnostic class to take in input_ids and labels and spit out tokens"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def batch_tokenize(self, texts):
        """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
        tokenized = [
            self.tokenizer(
                prompt, return_tensors="pt", padding="longest", truncation=True
            ).input_ids
            for prompt in texts
        ]
        return tokenized

    def construct_dataset(self, input_data):
        prompts = [val["prompt"] for val in input_data]
        tokenized_input_ids = self.batch_tokenize(prompts)
        labels = [val["completion"] for val in input_data]
        tokenized_labels = self.batch_tokenize(labels)
        return TuneDataset(tokenized_input_ids, tokenized_labels)

class CausalDatasetBuilder(DatasetBuilder):
    """Builds generative dataset for Causal LM."""

    def __init__(self, tokenizer, train_on_prompt=True):
        super().__init__(tokenizer)
        self.train_on_prompt = train_on_prompt

    def construct_dataset(self, input_data):
        labels = [val["prompt"] + "\n" + val["completion"] + self.tokenizer.eos_token for val in input_data]
        input_ids = [val.squeeze() for val in self.batch_tokenize(labels)]
        labels = copy.deepcopy(input_ids)
        if self.train_on_prompt:
            return TuneDataset(input_ids, labels)
        # masking prompt
        prompts = [val["prompt"] for val in input_data]
        tokenized_prompts = self.batch_tokenize(prompts)
        prompt_lens = [val.shape[1] for val in tokenized_prompts]

        for label, source_len in zip(labels, prompt_lens):
            label[:source_len] = IGNORE_INDEX
        return TuneDataset(input_ids, labels)

class TuneDataset(Dataset):
    """Dead simple torch dataset wrapper. Attention masks are created in collator"""

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SequenceDataCollator():
    """Collate examples for dynamic batch construction in supervised fine-tuning."""

    def __init__(self, tokenizer, multiple_of=None):
        self.tokenizer = tokenizer
        self.multiple_of = multiple_of
        self.cache_count = 0

    def pad_to_multiple(self, tensor, value):
        # taking advantage of tensor cores, perhaps
        multiple = self.multiple_of
        target_length = (tensor.size(0) + multiple - 1) // multiple * multiple
        return torch.nn.functional.pad(
            tensor, (0, target_length - tensor.size(0)), value=value
        )

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        if self.multiple_of:
            input_ids = [
                self.pad_to_multiple(val, self.tokenizer.pad_token_id)
                for val in input_ids
            ]
            labels = [self.pad_to_multiple(val, -100) for val in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )  # -100 tells torch to ignore these tokens in loss computation.

        #print(f"rank: {os.environ['RANK']}, cur memory: {torch.cuda.memory_allocated()}, max allocated: {torch.cuda.max_memory_allocated()}, peak memory: {torch.cuda.max_memory_reserved()}")
        if self.cache_count < 1:
            torch.cuda.empty_cache()
            #print(f"rank: {os.environ['RANK']} emptying cache ")
            self.cache_count += 1

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_data(path: Path) -> List[dict]:
    if path.suffix == ".json":
        return load_json(path)
    elif path.suffix == ".jsonl":
        return load_jsonl(path)
    else:
        raise Exception(
            f"file type {path} not supported. Currently supported types are json, jsonl"
        )


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            json_object = json.loads(line)
            data.append(json_object)
    return data


def load_json(path):
    """Loads a single json blob"""
    with open(path, "r") as f:
        data = json.load(f)
    return data

        
def train():
    print("Loading model...")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = load_tokenizer(model_args.model_name_or_path, training_args.model_max_length)

    print(f"Loading dataset {data_args.data_path}...")
    print(data_args.data_path)
    train_data = load_data(Path(data_args.data_path))
    p = CausalDatasetBuilder(tokenizer)
    train_dataset = p.construct_dataset(train_data)

    data_collator =SequenceDataCollator(tokenizer, 8)
    trainer = Trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=None,
                    data_collator=data_collator)


    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()

