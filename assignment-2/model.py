import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy
from tqdm import tqdm
class POSTagger(nn.Module):
    def __init__(self,max_seq_len,embeddings,hidden_dim,n_layers,tagset_size,device="cuda"):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_labels = tagset_size
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(embeddings,padding_idx=0)
        self.lstm = nn.LSTM(input_size=embeddings.size()[1], hidden_size= self.hidden_dim , num_layers=n_layers)
        self.hidden2tag = nn.Linear(self.hidden_dim,self.num_labels)
        self.device = device
        self.to(device)
        
    def forward(self,input_seq):
        input_seq = input_seq.to(self.device)
        embed_out =self.embeddings(input_seq)
        lstm_out,_ = self.lstm(embed_out)
        logits = self.hidden2tag(lstm_out)
        return logits
    
    def evaluate(self,loader):
        self.eval()
        true_labels = []
        pred_labels = []
        for i, data in enumerate(loader):
            x,y = data
            logits = self.forward(x)
            pred_label=torch.argmax(logits, dim=-1).cpu().numpy()
            batch_size, _ = x.shape
            for j in range(batch_size):
                tags = y[j]
                pred = pred_label[j]
                for k in range(len(tags)):
                    if tags[k] != 0:
                        true_labels.append(tags[k])
                        pred_labels.append(pred[k])
        acc = accuracy(true_labels, pred_labels)  
        return acc ,true_labels ,pred_labels          

    def run_training(self,train_loader,dev_loader,epochs=100,learning_rate=5e-4,eval_every=5):
        if str(self.device) == 'cpu':
            print("Training only supported in GPU environment")
            return
        torch.cuda.empty_cache()
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        # print model training parameters
        print("Training model with parameters | EPOCHS: {} LEARNING RATE: {}".format(epochs,learning_rate))
        for epoch in tqdm(range(epochs)):
            self.train()
            total_loss = 0
            for i, data in enumerate(train_loader):
                x,y=data
                self.zero_grad()
                logits = self.forward(x)
                labels = torch.LongTensor(y).to(self.device)
                loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
                total_loss += loss
                loss.backward()
                optimizer.step()
            # print("Epoch {} | Loss: {}".format(epoch, total_loss))
            # if epoch % eval_every == eval_every-1:
            #     acc,_,_ = self.evaluate(dev_loader)
            #     print("Epoch {} | Accuracy: {}".format(epoch, acc))
        acc_train,_,_ = self.evaluate(train_loader)
        acc_val,true_labels,pred_labels = self.evaluate(dev_loader)
        print("# Model : Training Accuracy : {} Validation Accuracy: {} #".format(acc_train,acc_val))  
    
    def predict(self,data):
        x = torch.LongTensor(data)
        self.eval()
        predictions = []
        logits = self.forward(x)
        pred_label=torch.argmax(logits, dim=-1).cpu().numpy()
        batch_size, _ = x.shape
        for j in range(batch_size):
            labels=[]
            for k in range(len(x[j])):
                if x[j][k] != 0:
                    labels.append(pred_label[j][k])
            predictions.append(labels)
        return predictions

    def summary(self):
        print ("Model Summary :")
        print('=====================================================================================================')
        print(self)
        print('=====================================================================================================')

    def save(self,path):
        torch.save(self.state_dict(), path)

    def load(self,path):
        self.load_state_dict(torch.load(path))



class ImprovedPOSTagger(POSTagger):
    def __init__(self,max_seq_len,embeddings,hidden_dim,n_layers,tagset_size,dropout=0.8,bidirectional=True,device="cuda"):
        super().__init__(max_seq_len,embeddings,hidden_dim,n_layers,tagset_size,device)
        self.lstm = nn.LSTM(input_size=embeddings.size()[1], hidden_size=hidden_dim, dropout=dropout, num_layers=n_layers, bidirectional=bidirectional)
        self.hidden2tag = nn.Linear(self.hidden_dim*2,self.num_labels)
        self.to(device)