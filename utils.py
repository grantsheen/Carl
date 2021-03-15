import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset import ConversationDataset

def load_dataset(args, tokenizer, filepath):
    """ Create Conversation Dataset from filepath.
    """
    df = pd.read_csv(filepath).replace(np.nan, '', regex=True)
    return ConversationDataset(tokenizer, df, int(args['--block-size']))

def evaluate_ppl(model, dataloader, device, batch_size):
    """ Calculate loss and perplexity of model on validation or test sentences.
    """
    was_training = model.training
    model.eval()

    total_loss, total_predictions = 0.0, 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            # prepare data
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # calculate loss (assumption: avg. per prediction)
            loss, logits, _ = model(inputs, labels=labels)
            
            # update counts
            num_predictions = batch_size * logits.shape[1]
            total_loss += loss * num_predictions
            total_predictions += num_predictions

        loss = total_loss / total_predictions
        ppl = torch.exp(loss)

    if was_training:
        model.train()

    return loss, ppl
