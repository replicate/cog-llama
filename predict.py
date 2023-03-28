
import json
import argparse

from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from torch.utils.data import Dataset
import torch
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
        tokenized = [
            self.tokenizer(
                prompt, 
                return_tensors='pt',
                padding='longest',
                ).input_ids
            for prompt in texts
        ]
        # if labels:
        #     tokenized = [val.input_ids for val in tokenized]

        #     # # change pad token id to -100 in labels s.t. loss function ignores it. 
        #     # def _ignore_pad(sequence):
        #     #     sequence[sequence == self.tokenizer.pad_token_id] = -100
        #     # _ = map(_ignore_pad, tokenized)
        return tokenized

    def make_prompt(self, input_row):
        if 'input' in input_row.keys():
            return self.prompt_dict['prompt_input'].format_map(input_row)
        return self.prompt_dict['prompt_no_input'].format_map(input_row)
    
    def construct_dataset(self, input_data):
        prompts = [self.make_prompt(val) for val in input_data]
        tokenized_input_ids = self.batch_tokenize(prompts) 
        labels = [val['output'] for val in input_data]
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
    

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key][0] for instance in instances] for key in ("input_ids", "labels"))
        #pdb.set_trace()
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        


def load_json(path):
    """Loads a single json blob"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data
    

def load_model(model_name_or_path):
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir="pretrained_weights")
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, cache_dir="pretrained_weights")
    return model, tokenizer


def train(args): 
    print('loading model')
    model, tokenizer = load_model(args.model_name_or_path)
    print('loading dataset')
    dataset = load_json(args.data_path)
    p = Preprocessor(tokenizer)
    train_data = p.construct_dataset(dataset)
    print('training')
    print(train_data[0])
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args = TrainingArguments(
            tf32=True,
            output_dir=args.output_path,
            per_device_train_batch_size=args.batch_size, 
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            save_strategy='epoch',
            save_total_limit=3,
            logging_steps=1,
            lr_scheduler_type='cosine',
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fsdp='full_shard'
            ),
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a language model on a text dataset")

    parser.add_argument("--data_path", type=str, required=True, help="Path to the text dataset")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The model class to fine-tune on HF or as a local path (e.g. 'google/flan-t5-xxl'")
    parser.add_argument("--output_path", type=str, required=True, help="Output path on disk to store your model")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Number of warmup steps for the learning rate scheduler")

    args = parser.parse_args()
    train(args)
