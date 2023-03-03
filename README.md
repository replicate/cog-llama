# FLAN-T5 Cog model

This is an implementation of [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights 

Then you can generate text based on input prompts:

    cog predict -i prompt="Q: Answer the following yes/no question by reasoning step-by-step. Can a dog drive a car?"
