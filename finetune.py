import torch
import os
import sys
import math
import time
from typing import Dict
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from utils import load_dataset, evaluate_ppl
from tqdm import tqdm

from transformers import (
    AutoConfig, 
    AutoModelWithLMHead, 
    AutoTokenizer,
)

def finetune(args: Dict):
    """ Finetune the DialoGPT Model.
    """
    tb = SummaryWriter()
    batch_size, eval_batch_size = int(args['--batch-size']), int(args['--eval-batch-size'])
    log_every, valid_niter = int(args['--log-every']), int(args['--valid-niter'])

    # load pretrained model
    config = AutoConfig.from_pretrained(args['--model-name'])
    tokenizer = AutoTokenizer.from_pretrained(args['--model-name'], pad_token='<pad>')
    model = AutoModelWithLMHead.from_pretrained(
        args['--model-name'],
        config=config
    )
    model.train()

    # setup device
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device)
    model = model.to(device)

    def pad(examples):
        """ Pad examples within a batch
        """
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    # training data
    train_dataset = load_dataset(args, tokenizer, args['--train-data'])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=pad,
        drop_last=True
    )

    # validation data
    val_dataset = load_dataset(args, tokenizer, args['--val-data'])
    val_sampler = RandomSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=eval_batch_size,
        collate_fn=pad,
        drop_last=True
    )
    
    # Adam optimizer
    optimizer = Adam(model.parameters(), lr=float(args['--lr']))

    # initialize parameters
    global_step = patience = num_trial = 0
    logging_loss, logging_predictions = 0.0, 0
    best_ppl = float('inf')
    output_dir = args['--output-dir']

    # finetune model
    print('begin finetuning!')
    for epoch in range(1, int(args['--max-epoch']) + 1):
        for _, batch in enumerate(tqdm(train_dataloader)):
            # prepare data
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # calculate loss (assumption: avg. per prediction)
            loss, logits, _ = model(inputs, labels=labels)

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args['--clip-grad']))
            optimizer.step()

            # update counts
            num_predictions = batch_size * logits.shape[1]
            batch_loss = loss * num_predictions
            logging_loss += batch_loss
            logging_predictions += num_predictions
            global_step += 1

            # logging
            if global_step % log_every == 0:
                avg_loss = logging_loss / logging_predictions
                tb.add_scalar('loss', avg_loss, global_step)

                avg_ppl = torch.exp(avg_loss)
                tb.add_scalar('ppl', avg_ppl, global_step)

                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f' % (epoch, global_step, avg_loss, avg_ppl))

                logging_loss, logging_predictions = 0.0, 0
                                                                           
            # perform validation
            if global_step % valid_niter == 0:
                print('begin validation ...')
                val_loss, val_ppl = evaluate_ppl(model, val_dataloader, device, eval_batch_size)
                print('validation: iter %d, val. loss %.2f, val. ppl %.2f' % (global_step, val_loss, val_ppl))

                tb.add_scalar('val. loss', val_loss, global_step)
                tb.add_scalar('val. ppl', val_ppl, global_step)  
                
                if val_ppl < best_ppl:
                    tb.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)             
                    best_ppl = val_ppl
                    patience = 0

                    # save current best model and optimizer
                    print('save the current best model to [%s]' % output_dir)
                    os.makedirs(output_dir, exist_ok=True) 
                    model.save_pretrained(output_dir)
                    torch.save(optimizer.state_dict(), output_dir + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience)
                    
                    # max patience reached
                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!')
                            tb.close()
                            exit(0)

                        # decay lr
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        tb.add_scalar('lr', lr, global_step)
                        print('decay learning rate to %f' % lr)

                        # restore model from previous best checkpoint
                        print('load previous best model')
                        model = AutoModelWithLMHead.from_pretrained(output_dir)
                        model = model.to(device)
                        
                        # restore optimizer from previous best checkpoint
                        print('restore optimizer parameters')
                        optimizer.load_state_dict(torch.load(output_dir + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0
               
    print('reached maximum number of epochs!')
    tb.close() 
    return 