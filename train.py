import json
import argparse
import os
from typing import Optional

from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset
import torch
from transformers import Trainer, TrainingArguments

TOKENIZER_NAME = "google/flan-t5-base" # this is a hack
MODEL_OUT_PATH = "tuned_weights"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
os.makedirs(MODEL_OUT_PATH, exist_ok=True)


class Preprocessor:
    """Simple class to parse alpaca data and return dataset. Very dataset specific, not trying to be anything else."""

    def __init__(self, tokenizer):
        self.prompt_dict = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }
        self.tokenizer = tokenizer

    def batch_tokenize(self, texts):
        """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
        tokenized = [
            self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
            ).input_ids
            for prompt in texts
        ]
        return tokenized

    def make_prompt(self, input_row):
        if "input" in input_row.keys():
            return self.prompt_dict["prompt_input"].format_map(input_row)
        return self.prompt_dict["prompt_no_input"].format_map(input_row)

    def construct_dataset(self, input_data):
        prompts = [self.make_prompt(val) for val in input_data]
        tokenized_input_ids = self.batch_tokenize(prompts)
        labels = [val["output"] for val in input_data]
        tokenized_labels = self.batch_tokenize(labels)
        return TuneDataset(tokenized_input_ids, tokenized_labels)


class TuneDataset(Dataset):
    """Dead simple torch dataset wrapper. Attention masks are created in collator"""

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class CustomDataCollatorSeq2Seq:
    """Collate examples for dynamic batch construction in supervised fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key][0] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )  # -100 tells torch to ignore these tokens in loss computation.
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_json(path):
    """Loads a single json blob"""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_json_string(data):
    return json.loads(data)


def load_model(model_name_or_path):
    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path, cache_dir="pretrained_weights"
    )
    tokenizer = T5Tokenizer.from_pretrained(
        model_name_or_path, cache_dir="pretrained_weights"
    )
    return model, tokenizer


# todo - eval stuff
def train(
    model_name_or_path=str,
    data=str,
    train_batch_size: int = 8,
    gradient_accumulation_steps: int = 8,
    lr_scheduler_type: str = "cosine",
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    num_train_epochs: int = 1,
    logging_steps: int = 1,
    **kwargs
):
    print("loading model")
    model, tokenizer = load_model(model_name_or_path)
    print("loading dataset")
    dataset = load_json_string(data)
    p = Preprocessor(tokenizer)
    train_data = p.construct_dataset(dataset)
    print("training")
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args=TrainingArguments(
            tf32=True,
            output_dir=CHECKPOINT_DIR,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy=SAVE_STRATEGY,
            save_total_limit=1,
            logging_steps=logging_steps,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            **kwargs
        ),
        data_collator=CustomDataCollatorSeq2Seq(tokenizer),
    )
    trainer.train()
    trainer.save_model(MODEL_OUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on a text dataset"
    )

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the text dataset"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The model class to fine-tune on HF or as a local path (e.g. 'google/flan-t5-xxl'",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Number of warmup steps for the learning rate scheduler",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        required=False,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )

    args = vars(parser.parse_args())
    # hack hack hack
    data = load_json(args.pop("data_path"))
    data = json.dumps(data)
    train(data=data, **args)
