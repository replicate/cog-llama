# LLaMA Cog template 🦙

LLaMA is a [new open-source language model from Meta Research](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) that performs as well as closed-source models. 

Similar to Stable Diffusion, this has created a wealth of experiments and innovation. [As Simon Willison articulated](https://simonwillison.net/2023/Mar/11/llama/), it's easy to run on your own hardware, large enough to be useful, and open-source enough to be tinkered with.

This is a guide to running LLaMA using in the cloud using Replicate. You'll use the [Cog](https://github.com/replicate/cog) command-line tool to package the model and push it to Replicate as a web interface and API.

This model can be used to run the `7B` version of LLaMA and it also works with fine-tuned models.

**Note: LLaMA is for research purposes only. It is not intended for commercial use.**

## Prerequisites

- **LLaMA weights**. The weights for LLaMA have not yet been released publicly. To apply for access, fill out [this Meta Research form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform).
- **GPU machine**. You'll need a Linux machine with an NVIDIA GPU attached and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. If you don't already have access to a machine with a GPU, check out our [guide to getting a GPU machine](https://replicate.com/docs/guides/get-a-gpu-machine).
- **Docker**. You'll be using the [Cog](https://github.com/replicate/cog) command-line tool to build and push a model. Cog uses Docker to create containers for models.

## Step 0: Install Cog

First, [install Cog](https://github.com/replicate/cog#install):

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

## Step 1: Set up weights

Replicate currently supports the `7B` model size.

Put your downloaded weights in a folder called `unconverted-weights`. The folder hierarchy should look something like this: 

```
unconverted-weights
├── 7B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
├── tokenizer.model
└── tokenizer_checklist.chk
```

Convert the weights from a PyTorch checkpoint to a transformers-compatible format using the this command:

```
cog run python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir unconverted-weights --model_size 7B --output_dir weights
```

You final directory structure should look like this:

```
weights
├── config.json
├── generation_config.json
├── pytorch_model-00001-of-00002.bin
├── pytorch_model-00002-of-00002.bin
├── pytorch_model.bin.index.json
├── special_tokens_map.json
├── tokenizer.model
└── tokenizer_config.json
```

Once you've done this, you should uncomment `unconverted-weights` in your `.dockerignore` file. This ensures that `unconverted-weights` aren't built into the resulting cog image.

## Step 2: Run the model

You can run the model locally to test it:

```
cog predict -i prompt="Simply put, the theory of relativity states that"
```

LLaMA is not fine-tuned to answer questions. You should construct your prompt so that the expected answer is the natural continuation of your prompt. 

Here are a few examples from the [LLaMA FAQ](https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/FAQ.md#2-generations-are-bad):

- Do not prompt with "What is the meaning of life? Be concise and do not repeat yourself." but with "I believe the meaning of life is"
- Do not prompt with "Explain the theory of relativity." but with "Simply put, the theory of relativity states that"
- Do not prompt with "Ten easy steps to build a website..." but with "Building a website can be done in 10 simple steps:\n"

## Step 3: Create a model on Replicate

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model.

Make sure to specify "private" to keep the model private.

## Step 4: Configure the model to run on A100 GPUs

Replicate supports running models on a variety of GPUs. The default GPU type is a T4, but for best performance you'll want to configure your model to run on an A100.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 5: Push the model to Replicate

Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 3:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)


## Step 6: Run the model on Replicate

Now that you've pushed the model to Replicate, you can run it from the website or with an API.

To use your model in the browser, go to your model page.

To use your model with an API, click on the "API" tab on your model page. You'll see commands to run the model with cURL, Python, etc.

To learn more about how to use Replicate, [check out our documentation](https://replicate.com/docs).