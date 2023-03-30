This project illustrates how to fine-tune a model within Cog on Replicate. 

Presently it's using the Alpaca dataset to fine-tune a flan-T5 model, but you'll notice that the actual training and serving scripts are model agnostic; 
process_data.py is provided only as an example. 

All the script requires is an input dataset consisting of a JSON list where each example has a 'prompt' and 'output' field. The model will be fine-tuned
to produce 'output' given a prompt. 
