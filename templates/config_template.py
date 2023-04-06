from transformers import T5Tokenizer

HUGGINGFACE_MODEL_NAME = "{{model_name}}"


def load_tokenizer():
    """Same tokenizer, agnostic from tensorized weights/etc"""
    return T5Tokenizer.from_pretrained(
        HUGGINGFACE_MODEL_NAME, cache_dir="pretrained_weights"
    )
