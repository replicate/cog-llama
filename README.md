# FLAN-T5 Cog model

This is an implementation of [LLaMA] as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

If you have LLaMA weights, here's how to convert them. Make sure to specify model_size properly.

    python /root/.pyenv/versions/3.10.10/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir /src/llamas/ --model_size 7B --output_dir weights_conv

If your weights are already converted, great! Just place them in `weights_conv`.Then you can generate text based on input prompts:

    cog predict -i prompt="How do you play the accordion?"
