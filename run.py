"""
CS 224N 2020-21: Final Project
run.py: Run Script for Empathetic Chatbot
Grant Sheen <gsheen@stanford.edu>

Usage: 
    run.py finetune [options]
    run.py test [options]
    run.py demo [options]

Options: 
    -h --help                               show this screen.
    --cuda                                  use GPU
    --dialoGPT                              use DialoGPT
    --train-data=<file>                     train data file [default: train.csv]
    --val-data=<file>                       val data file [default: valid.csv]
    --test-data=<file>                      test data file [default: test.csv]
    --model-name=<file>                     pretrained model name or path [default: microsoft/DialoGPT-small]
    --model-save-path=<file>                model save path [default: model.bin]
    --block-size=<int>                      block size [default: 512]
    --lr=<float>                            learning rate [default: 0.001]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --adam-epsilon=<float>                  epsilon for Adam optimizer [default: 1e-8]
    --batch-size=<int>                      batch size [default: 16]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --seed=<int>                            seed [default: 42]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --beam-size=<int>                       beam size [default: 5]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --num-lines=<int>                       number of lines to chat in demo [default: 10]
"""
from docopt import docopt
import numpy as np
import torch

from finetune import finetune
from test import test
from demo import demo

def main():
    """ Main function.
    """
    args = docopt(__doc__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if args['finetune']:
        finetune(args)
    elif args['test']:
        test(args)
    elif args['demo']: 
        demo(args)
    else:
        raise RuntimeError('invalid run mode')

if __name__ == "__main__":
    main()