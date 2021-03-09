import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from dataset import ConversationDataset

def pad(examples):
    """ Pad examples within a batch
    """
    return pad_sequence(examples, batch_first=True)

def load_dataset(args, tokenizer, filepath):
    """ Create Conversation Dataset from filepath.
    """
    df = pd.read_csv(filepath).replace(np.nan, '', regex=True)
    return ConversationDataset(tokenizer, df, int(args['--block-size']))

def evaluate_ppl(model, dataloader, device, batch_size):
    """ Calculate perplexity of model on validation or test sentences.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tgt_words = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            # prepare data
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # calculate loss (avg.)
            loss = model(inputs, labels=labels)[0]
            
            # update counts
            batch_loss = loss * batch_size
            total_loss += batch_loss
            words_to_predict = sum(len(s) for s in labels)
            total_tgt_words += words_to_predict

        ppl = np.exp(total_loss / total_tgt_words)

    if was_training:
        model.train()

    return ppl