import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class SentimentClassifier(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,elmo_embeddings,elmo_l1,elmo_l2,dropout=0.2):
        super(SentimentClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(elmo_embeddings)
        self.embedding.weight = nn.Parameter(self.embedding.weight, requires_grad=True)
        self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]), requires_grad=True)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim*2)
        self.lstm1 = elmo_l1
        self.lstm2 = elmo_l2
        self.linear2 = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input_data):
        embeds = self.embedding(input_data)
        embeds_change = self.linear1(embeds)
        hidden1, _ = self.lstm1(embeds)
        hidden2, _ = self.lstm2(hidden1)
        elmo_embed = (self.weights[0]*hidden1 + self.weights[1]*hidden2 
                      + self.weights[2]*embeds_change)/(self.weights[0]+self.weights[1]+self.weights[2])
        elmo_embed_max = torch.max(elmo_embed, dim=1)[0]
        elmo_embed_max_drop = self.dropout(elmo_embed_max)
        linear_out = self.linear2(elmo_embed_max_drop)
        return torch.sigmoid(linear_out)