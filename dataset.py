import torch
from torch.utils.data import DataLoader
from config import MAX_SEQ_LEN,BATCH_SIZE,TRAIN_ENCODED_TEXTES,TEST_ENCODED_TEXTES

class WikiTextDataset:
    def __init__(self,all_tokens,block_size):
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.block_size = block_size
        self.num_samples = (len(all_tokens) - 1) // self.block_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,index):
        
        start = index * self.block_size
        end = start + self.block_size
        
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        
        return x,y

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_tokens = torch.load(TRAIN_ENCODED_TEXTES,weights_only=True)
train_dataset = WikiTextDataset(train_tokens,MAX_SEQ_LEN)    

train_dataloader = DataLoader(
    train_dataset,
    BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    pin_memory_device=device
)

test_tokens = torch.load(TEST_ENCODED_TEXTES,weights_only=True)
test_dataset = WikiTextDataset(test_tokens,MAX_SEQ_LEN)

test_dataloader = DataLoader(
    test_dataset,
    BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    pin_memory_device=device
)  