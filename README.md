# LLaMA Cog template

A template to run LLaMA in [Cog](https://github.com/replicate/cog) with Transformers.

Put your weights in a folder `weights_conv/`. This folder should have these sub-folders:

* `weights_conv/llama-7b/` – checkpoints
* `weights_conv/tokenizer/` – tokenizer

If you have LLaMA weights you can convert them with this command. Make sure to specify model_size properly.

    python /root/.pyenv/versions/3.10.10/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir /src/llamas/ --model_size 7B --output_dir weights_conv

First, [install Cog](https://github.com/replicate/cog#install).

You can run the model locally to test it:

    cog predict -i prompt="How do you play the accordion?"

Log in to Replicate:

    cog login

To push it to Replicate, run:

    cog push r8.im/username/modelname

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)
