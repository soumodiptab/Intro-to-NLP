import torch
from torch.utils.data import Dataset, DataLoader

class POSDataSet(Dataset):
    def __init__(self, x, y):
        self.sent = torch.LongTensor(x)
        self.sent_tags = torch.LongTensor(y)
    
    def __getitem__(self, idx):
        return self.sent[idx], self.sent_tags[idx]
    
    def __len__(self):
        return len(self.sent)
    
    def get_dataloader(self, batch_size,shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
