from collections import OrderedDict
import logging
import re
import time
import os
from transformers import LlamaTokenizer, AutoConfig, LlamaForCausalLM
import torch
import subprocess
from subprocess import DEVNULL, STDOUT
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor

from subclass import YieldingLlama

DEFAULT_MODEL_NAME = "llama_weights/llama-7b"  # path from which we pull weights when there's no COG_WEIGHTS environment variable
TOKENIZER_NAME = "llama_weights/tokenizer"
CONFIG_LOCATION = "llama_weights/llama-7b"

LOCAL_PATH = "/src/llama.tensors"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def load_tokenizer():
    """Same tokenizer, agnostic from tensorized weights/etc"""
    tok = LlamaTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir="pretrained_weights")
    tok.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tok


def https_to_gs_path(weights):    
    """Translates from replicate.delivery -> gs://, no-op otherwise"""
    weights = str(weights)
    pattern = r'https://pbxt\.replicate\.delivery/([^/]+/[^/]+)'
    match = re.search(pattern, weights)
    if match:
        weights = f"gs://replicate-files/{match.group(1)}"
    return weights


def maybe_download(path=DEFAULT_MODEL_NAME, output_path=LOCAL_PATH):
    """uses gcloud storage to pull replicate.delivery or gs:// files to a local directory"""
    st = time.time()    
    path = https_to_gs_path(path)
    print(f"Downloading {path} to {output_path}")
    if path.startswith("gs://") and not os.path.exists(output_path):
        subprocess.check_call(["/gc/google-cloud-sdk/bin/gcloud", "storage", "cp", path, output_path])
        print(f"Downloaded in {time.time() - st}")
        return output_path
    elif os.path.exists(output_path):
        return output_path
    return path


def load_tensorizer(
    weights, plaid_mode: bool = True, cls: LlamaForCausalLM = YieldingLlama
):
    st = time.time()
    weights = https_to_gs_path(weights)
    local_weights = maybe_download(path=weights, output_path=LOCAL_PATH)

    config = AutoConfig.from_pretrained(CONFIG_LOCATION)

    logging.disable(logging.WARN)
    model = no_init_or_tensor(
        lambda: cls.from_pretrained(
            None, config=config, state_dict=OrderedDict(), torch_dtype=torch.float16
        )
    )
    logging.disable(logging.NOTSET)

    des = TensorDeserializer(local_weights, plaid_mode=plaid_mode)
    des.load_into_module(model)
    print(f"weights loaded in {time.time() - st}")
    return model
