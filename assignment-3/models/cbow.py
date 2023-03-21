import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CBOW(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super(CBOW,self).__init__()
        self.embedding_dim = vocab_size
        self.embedding_size = embedding_size
        self.embeddings_target = nn.Embedding(vocab_size,embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size,embedding_size)
        self.log_sigmoid = nn.LogSigmoid()
        self.__init__weights()
    
    def __init__weights(self):
        # Xavier initialization
        initrange = (2.0 / (self.embedding_dim + self.embedding_size)) ** 0.5 
        self.embeddings_target.weight.data.uniform_(-initrange, initrange)
        self.embeddings_context.weight.data.uniform_(-0, 0)

    def forward(self,inputs):
        pass

    def trainer(self,epochs,lr,batch_size,dataset:Dataset):
        dataloader = dataset.get_batches(batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        

    def save_embeddings(self,filepath):
        pass

    def save_model(self,filepath):
        pass

    def load_model(self,filepath):
        pass
    
    def predict(self,inputs):
        return self.embeddings_target(inputs)