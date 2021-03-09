import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import Dataset

def tokenize(row, tokenizer):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list([tokenizer.encode(sentence) + [tokenizer.eos_token_id] for sentence in row if sentence])
    conv = flatten(conv)
    return conv

class ConversationDataset(Dataset):
    def __init__(self, tokenizer, df, block_size):
        self.examples = []
        for _, row in df.iterrows():
            conv = tokenize(row, tokenizer)
            if len(conv) > block_size:
                conv = conv[:block_size]
            self.examples.append(conv)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)



