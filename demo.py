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
        tokenizer = AutoTokenizer.from_pretrained(pretrained, pad_token='00000')
        name = 'Carl'

    for step in range(int(args['--num-lines'])):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generates a response based on chat history 
        if args['--dialoGPT']:
            chat_history_ids = model.generate(bot_input_ids, 
                max_length=1000,
                min_length=10,
                num_beams=3,
                temperature=0.8,
                length_penalty=2,
                no_repeat_ngram_size=3, 
                top_k=30, 
                top_p=0.8,  
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, 
            )
        else:
            chat_history_ids = model.generate(
                bot_input_ids, 
                max_length=1000,
                min_length=10,
                num_beams=3,
                temperature=0.8,
                length_penalty=2,
                no_repeat_ngram_size=3, 
                top_k=30, 
                top_p=0.8,  
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # pretty print last ouput tokens from bot
        print("{}: {}".format(name, tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
