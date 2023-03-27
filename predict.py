
import json
import argparse

from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

# Would be good to add some sort of "tuner" entity here s.t. we're not just running 

# Fill in with dataset module which can package up dataset. 
# broadly, we want to define some sort of object which can iterate over the dataset and return an input and also labels. 
# The simplest way to do this would be to define just a prompt templater object

class Preprocessor:
    """Simple class to parse dataset. Very dataset specific, not trying to be anything else."""

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
            )
        }
        self.tokenizer = tokenizer

    def batch_tokenize(self, texts, labels=False):
        """Tokenizes text. Presently pads all inputs to length of longest sequence."""
        # TODO: experiment with padding strategies
        lens = [self.tokenizer(tokens).input_ids for tokens in texts]
        max_len = max(lens)

        tokenized = [
            self.tokenizer(
                prompt, 
                return_tensors='pt',
                padding='max_length',
                max_length=max_len,
                )
            for prompt in texts
        ]
        if labels:
            tokenized = [val.input_ids for val in tokenized]

            # change pad token id to -100 in labels s.t. loss function ignores it. 
            def _ignore_pad(sequence):
                sequence[sequence == self.tokenizer.pad_token_id] = -100
            _ = map(_ignore_pad, tokenized)
        return tokenized

    def make_prompt(self, input_row):
        if 'input' in input_row.keys():
            return self.prompt_dict['prompt_input'].format_map(input_row)
        return self.prompt_dict['prompt_no_input'].format_map(input_row)
    
    def construct_dataset(self, input_data):
        prompts = [self.make_prompt(val) for val in input_data]
        tokenized_input_ids = self.batch_tokenize(prompts)
        labels = [val['label'] for val in input_data]
        tokenized_labels = self.batch_tokenize(labels, labels=True)
        return TuneDataset(tokenized_input_ids, tokenized_labels)


class TuneDataset(Dataset):

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        


def load_json(path):
    """Loads a single json blob"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data
    

def load_model(model_name_or_path):
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def train(args): 
    print('loading model')
    model, tokenizer = load_model(args.model_name_or_path)
    print('loading dataset')
    dataset = load_json(args.data_path)
    p = Preprocessor(tokenizer)
    train_data = p.construct_dataset(dataset)

    # TODO: if this is long AF then you can just write a dataCollator
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        args = TrainingArguments(
            bf16=True,
            output_dir='flan-t5-out'
            per_device_train_batch_size=4, 
            per_device_eval_batch_size=4
            gradient_accumulation_steps=8,
            save_strategy='epoch'
            save_total_limit=3,
            logging_steps=1,
            lr_scheduler_type='cosine',
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            )
        data_collator=
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a language model on a text dataset")

    parser.add_argument("--data_path", type=str, required=True, help="Path to the text dataset")
    parser.add_argument("--model_class", type=str, required=True, help="The model class to fine-tune on HF or as a local path (e.g. 'google/flan-t5-xxl'")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for the optimizer")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenized text")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for the learning rate scheduler")

    args = parser.parse_args()
    train(args)


# ok, so how do I pad these tokens? I need to go through and compute 

# so ignore index = pad token, but nothing else. 
# also yeah the alpaca code is set up for weird llama particulars
# this code, right here, is still useful if it only fine-tunes flan-t5 and you fully understand it. 
# so keep it scoped to that. 

# need to add attention_mask as well. 
