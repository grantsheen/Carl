import torch
from os import path
from torch.utils.data import DataLoader, RandomSampler
from utils import pad, load_dataset, evaluate_ppl
from transformers import (
    AutoConfig, 
    AutoModelWithLMHead,
    AutoTokenizer,
)

def test(args):
    """ Evaluates perplexity of the model on testing data.
    """
    config = AutoConfig.from_pretrained(args['--model-name'])
    model = AutoModelWithLMHead.from_pretrained(
        args['--model-name'],
        config=config
    )
    if not args['--dialoGPT']:
        # load finetuned model
        assert(path.exists(args['--model-save-path']))
        params = torch.load(args['--model-save-path'], map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])
    
    # setup device
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device)
    model = model.to(device)

    # testing data
    tokenizer = AutoTokenizer.from_pretrained(args['--model-name'])
    test_dataset = load_dataset(args, tokenizer, args['--test-data'])
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=int(args['--batch-size']),
        collate_fn=pad,
        drop_last=True
    )

    ppl = evaluate_ppl(model, test_dataloader, device, batch_size=128)

    if args['--dialoGPT']:
        print("Perplexity score of DialoGPT on testing data: %f" % ppl)
    else: 
        print("Perplexity score of finetuned model on testing data: %f" % ppl)

    return