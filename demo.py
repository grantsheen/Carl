import torch
from os import path
from typing import Dict
from transformers import (
    AutoConfig,
    AutoModelWithLMHead, 
    AutoTokenizer,
)

def demo(args):
    """ Chatbot demo!
    """
    pretrained = args['--model-name']
    finetuned = args['--output-dir']

    config = AutoConfig.from_pretrained(pretrained)
    model = AutoModelWithLMHead.from_pretrained(
        pretrained,
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained, pad_token='<pad>')

    if args['--dialoGPT']:
        name = 'DialoGPT'
    else:
        # load finetuned model
        assert(path.exists(finetuned))
        model = AutoModelWithLMHead.from_pretrained(finetuned)
        name = 'Roger'

    for step in range(int(args['--num-lines'])):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 500 tokens, 
        chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=500,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )

        # pretty print last ouput tokens from bot
        print("{}: {}".format(name, tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
