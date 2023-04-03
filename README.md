This project illustrates how to fine-tune a model within Cog on Replicate. 

Presently it's using the Alpaca dataset to fine-tune a flan-T5 model, but you'll notice that the actual training and serving scripts are model agnostic; 
process_data.py is provided only as an example. 

All the script requires is an input dataset consisting of a JSON list where each example has a 'prompt' and 'output' field. The model will be fine-tuned
to produce 'output' given a prompt. 

This project also has the ability to build and push a cog container for any of the FLAN family of models. Just run `cog run python render_template.py --model_name [small, base, large, xl, xxl, ul2]`, and then you can run all other `cog` commands with the appropriate model. 
