import torch
from os import path
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from utils import load_dataset, evaluate_ppl
from transformers import (
    AutoConfig, 
    AutoModelWithLMHead,
    AutoTokenizer,
)

def test(args):
    """ Evaluates perplexity of the model on testing data.
    """
    pretrained = args['--model-name']
    finetuned = args['--output-dir']
    eval_batch_size = int(args['--eval-batch-size'])

    def pad(examples):
        """ Pad examples within a batch
        """
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)    

    config = AutoConfig.from_pretrained(pretrained)
    model = AutoModelWithLMHead.from_pretrained(
        pretrained,
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained, pad_token='<pad>')

    # testing data
    test_dataset = load_dataset(args, tokenizer, args['--test-data'])
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=eval_batch_size,
        collate_fn=pad,
        drop_last=True
    )

    if not args['--dialoGPT']:
        # load finetuned model
        assert(path.exists(finetuned))
        model = AutoModelWithLMHead.from_pretrained(finetuned)
    
    # setup device
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device)
    model = model.to(device)

    loss, ppl = evaluate_ppl(model, test_dataloader, device, eval_batch_size)

    if args['--dialoGPT']:
        print("Loss of DialoGPT on testing data: %f" % loss)
        print("Perplexity score of DialoGPT on testing data: %f" % ppl)
    else: 
        print("Loss of finetuned model on testing data: %f" % loss)
        print("Perplexity score of finetuned model on testing data: %f" % ppl)

    return