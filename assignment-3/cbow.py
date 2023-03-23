import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
class CBOW_NEG(nn.Module):
    def __init__(self,vocab_size,embedding_size,model_path,embedding_path):
        super(CBOW_NEG,self).__init__()
        self.embedding_dim = vocab_size
        self.embedding_size = embedding_size
        self.embeddings_target = nn.Embedding(vocab_size,embedding_size,sparse=True)
        self.embeddings_context = nn.Embedding(vocab_size,embedding_size,sparse=True)
        self.model_path=model_path
        self.embedding_path=embedding_path
        self.log_sigmoid = nn.LogSigmoid()
        self.__init__weights()
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
    
    def __init__weights(self):
        # Xavier initialization
        initrange = (2.0 / (self.embedding_dim + self.embedding_size)) ** 0.5 
        self.embeddings_target.weight.data.uniform_(-initrange, initrange)
        self.embeddings_context.weight.data.uniform_(-0, 0)

    def forward(self,target,context,neg_samples):
        target_embedding = self.embeddings_target(target)
        context_embedding = self.embeddings_context(context)
        context_embedding = torch.mean(context_embedding, dim=1)
        neg_embedding = self.embeddings_context(neg_samples)
        pos_loss=torch.sum(target_embedding * context_embedding, dim=1)
        pos_loss = -self.log_sigmoid(pos_loss)
        neg_loss = torch.bmm(neg_embedding, context_embedding.unsqueeze(2)).squeeze()
        neg_loss = self.log_sigmoid(-neg_loss)
        neg_loss = -torch.sum(neg_loss, dim=1)
        loss = torch.mean(pos_loss + neg_loss)
        return loss

    def trainer(self,dataset:Dataset,batch_size=100,epochs=10,lr=0.001,print_every=2,checkpoint_every=5):
        if str(self.device) == 'cpu':
            print("Training only supported in GPU environment")
            return
        # Training with hyper - parameters 
        print('Training with hyper-parameters: lr: {}, batch_size: {}, embedding_size: {}'.format(lr,batch_size,self.embedding_size))
        torch.cuda.empty_cache()
        self.to(self.device)
        self.train()
        dataloader = dataset.get_batches(batch_size)
        optimizer = optim.SparseAdam(self.parameters(),lr=lr)
        steps =len(dataloader)
        for e in range(epochs):
            print('Epoch: {}/{}'.format(e+1,epochs))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,steps )
            avg_loss = 0
            for i,data in enumerate(tqdm(dataloader)):
                targets,contexts,neg_samples = data
                targets = targets.to(self.device)
                contexts = contexts.to(self.device)
                neg_samples = neg_samples.to(self.device)
                optimizer.zero_grad()
                loss = self.forward(targets,contexts,neg_samples)
                loss.backward()
                optimizer.step()
                scheduler.step()
                avg_loss+= loss.item()
            avg_loss = avg_loss/steps
            if i % print_every == 0:
                    print('Loss: {}'.format(avg_loss))
            if e % checkpoint_every == 0:
                # save model with hyperparameters
                print('x------------------Saving embeddings------------------x')
                model_rel_path = os.path.join(self.model_path,'cbow_neg_lr_{}_e_{}.pth'.format(lr,self.embedding_size))
                self.save_model(model_rel_path)
                embedding_path = os.path.join(self.embedding_path,'cbow_neg_embeddings_{}.txt'.format(self.embedding_size))
                self.save_embeddings(dataset.ind2vocab,embedding_path)

    def save_embeddings(self,id2word,filepath):
        embeddings = self.embeddings_target.weight.data.cpu().numpy()
        with open(filepath,'w') as f:
            f.write('{} {}\n'.format(len(embeddings),self.embedding_size))
            for i,e in enumerate(embeddings):
                f.write(id2word[i] + ' ' + ' '.join([str(v) for v in e]) + '\n')
                

    def save_model(self,filepath):
        torch.save(self.state_dict,filepath)

    def load_model(self,filepath):
        self.load_state_dict(torch.load(filepath))
    
    def predict(self,inputs):
        return self.embeddings_target(inputs)