import torch
import sys
import math
import time
from typing import Dict
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam
from utils import pad, load_dataset, evaluate_ppl
from tqdm import tqdm

from transformers import (
    AutoConfig, 
    AutoModelWithLMHead, 
    AutoTokenizer,
)

def finetune(args: Dict):
    """ Finetune the DialoGPT Model.
    """
    # load pretrained model
    config = AutoConfig.from_pretrained(args['--model-name'])
    tokenizer = AutoTokenizer.from_pretrained(args['--model-name'])
    model = AutoModelWithLMHead.from_pretrained(
        args['--model-name'],
        config=config
    )
    model.train()

    # setup device
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device)
    model = model.to(device)

    # training data
    train_dataset = load_dataset(args, tokenizer, args['--train-data'])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=int(args['--batch-size']),
        collate_fn=pad,
        drop_last=True
    )

    # validation data
    val_dataset = load_dataset(args, tokenizer, args['--val-data'])
    val_sampler = RandomSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=int(args['--batch-size']),
        collate_fn=pad,
        drop_last=True
    )

    # Adam optimizer
    optimizer = Adam(model.parameters(), lr=float(args['--lr']))

    # initialize parameters
    total_examples = report_examples = total_tgt_words = report_tgt_words = iter_count = patience = num_trial = 0
    batch_size, log_every, valid_niter = int(args['--batch-size']), int(args['--log-every']), int(args['--valid-niter'])
    model_save_path = args['--model-save-path']
    total_loss = report_loss = 0.0
    best_ppl = float('inf')
    train_time = begin_time = time.time()

    # finetune model
    print('begin finetuning!')
    for epoch in range(1, int(args['--max-epoch']) + 1):
        for _, batch in enumerate(tqdm(train_dataloader)):
            # prepare data
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # calculate loss (avg.)
            loss = model(inputs, labels=labels)[0]

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args['--clip-grad']))
            optimizer.step()

            # update counts
            batch_loss = loss * batch_size
            report_loss += batch_loss
            total_loss += batch_loss
            report_examples += batch_size
            total_examples += batch_size
            words_to_predict = sum(len(s) for s in labels)
            report_tgt_words += words_to_predict
            total_tgt_words += words_to_predict
            iter_count += 1

            # logging
            if iter_count % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'tot. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, iter_count,
                                                                                         report_loss / report_examples,
                                                                                         torch.exp(report_loss / report_tgt_words),
                                                                                         total_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time))
                train_time = time.time()
                report_loss = 0.0
                report_examples = report_tgt_words = 0                                                                        

            # perform validation
            if iter_count % valid_niter == 0:
                print('epoch %d, iter %d, tot. loss %.2f, tot. ppl %.2f tot. examples %d' % (epoch, iter_count,
                                                                                            total_loss / total_examples,
                                                                                            torch.exp(total_loss / total_tgt_words),
                                                                                            total_examples))
                total_loss = 0.0
                total_examples = total_tgt_words = 0

                print('begin validation ...')
                val_ppl = evaluate_ppl(model, val_dataloader, device, batch_size=128)
                print('validation: iter %d, val. ppl %f' % (iter_count, val_ppl))
                
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    patience = 0
                    print('save the current best model to [%s]' % model_save_path)
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience)
                    
                    # max patience reached
                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!')
                            exit(0)

                        # decay lr
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('decay learning rate to %f' % lr)

                        # restore model from previous best checkpoint
                        print('load previous best model')
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)
                        
                        # restore optimizer from previous best checkpoint
                        print('restore optimizer parameters')
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0
                
    print('reached maximum number of epochs!')
    return 