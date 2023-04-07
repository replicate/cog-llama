from transformers import LlamaTokenizer

DEFAULT_MODEL_NAME = "weights/llama-7b" # path from which we pull weights when there's no COG_WEIGHTS environment variable
TOKENIZER_NAME = "weights/tokenizer"
CONFIG_LOCATION = "weights/llama-7b"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def load_tokenizer():
    """Same tokenizer, agnostic from tensorized weights/etc"""
    tok = LlamaTokenizer.from_pretrained(
        TOKENIZER_NAME, cache_dir="pretrained_weights"
    )
    tok.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
        "pad_token": DEFAULT_PAD_TOKEN
    })
    return tok