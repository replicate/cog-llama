# LLaMA Cog model

This is an implementation of LLaMA as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

Put your weights in a folder `weights_conv/`.

If you have LLaMA weights you can convert them with this command. Make sure to specify model_size properly.

    python /root/.pyenv/versions/3.10.10/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir /src/llamas/ --model_size 7B --output_dir weights_conv

Now, you can run the model locally:

    cog predict -i prompt="How do you play the accordion?"

To push it to Replicate, run:

    cog push r8.im/username/modelname
