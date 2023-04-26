import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score
from elmo import ELMo
from matplotlib import pyplot as plt
import numpy as np



class ElmoTrainer:
    def __init__(self,epochs=20,lr=0.001,batch_size=50,print_every=1,device='cpu'):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.print_every = print_every
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.lowest_validation_loss = float('inf')
        self.training_loss_history = []
    
    def train(self,model : ELMo,model_save_path,train_data,validation_data):
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.training_loss_history = []
        model.to(self.device)
        for epoch in range(len(range(self.epochs))):
            model.train()
            train_loader = train_data.get_batches(self.batch_size)
            training_loss = 0
            for (forward_data,backward_data) in tqdm(train_loader):
                forward_data = forward_data.to(self.device)
                backward_data = backward_data.to(self.device)
                self.optimizer.zero_grad()
                output = model(backward_data)
                output = output.view(-1, model.vocab_size)
                target = forward_data.view(-1)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
            if epoch % self.print_every == 0:
                print('Training Loss : {}'.format(training_loss/len(train_loader)))
            self.__validate(model,model_save_path,validation_data)
            self.training_loss_history.append(training_loss/len(train_loader))

    def __validate(self,model:ELMo,model_save_path,validation_data):
        model.eval()
        model.to(self.device)
        validation_loader = validation_data.get_batches(self.batch_size)
        validation_loss = 0
        for i,(forward_data,backward_data) in tqdm(enumerate(validation_loader)):
            forward_data = forward_data.to(self.device)
            backward_data = backward_data.to(self.device)
            output = model(backward_data)
            output = output.view(-1, model.vocab_size)
            target = forward_data.view(-1)
            loss = self.criterion(output, target)
            validation_loss += loss.item()
        if validation_loss < self.lowest_validation_loss:
            self.lowest_validation_loss = validation_loss
            torch.save(model.state_dict(), model_save_path)
        print('Validation Loss : {}'.format(validation_loss/len(validation_loader)))

    def plot_loss(self):
        plt.plot(self.training_loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.show()

    def get_embeddings(self,model:ELMo):
        elmo_embeddings = list(model.parameters())[0].cpu().detach().numpy()
        return torch.FloatTensor(elmo_embeddings)


class SentTrainer:
    def __init__(self,epochs=20,lr=0.001,batch_size=50,print_every=1,device='cpu'):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.print_every = print_every
        self.device = device
        self.criterion = nn.MSELoss()
        self.lowest_validation_loss = float('inf')
        self.training_loss_history = []
        self.training_accuracy_history = []
        self.training_f1_score_history = []
        self.validation_loss_history = []
    
    def train(self,model : ELMo,model_save_path,train_data,validation_data):
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.training_loss_history = []
        model.to(self.device)
        for epoch in range(len(range(self.epochs))):
            model.train()
            train_loader = train_data.get_batches(self.batch_size)
            training_loss = 0
            training_acc = 0
            training_f1 = 0
            threshold = 0.5
            for (sent_data,target) in tqdm(train_loader):
                sent_data = sent_data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = model(sent_data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                true_labels = target.cpu().detach().numpy()
                pred_labels = output.cpu().detach().numpy()
                true_labels = (true_labels >= threshold).astype(np.int64)
                pred_labels = (pred_labels >= threshold).astype(np.int64)
                # get accuracy by comparing output with target and detaching from the layer
                training_acc+=accuracy_score(true_labels,pred_labels)
                training_f1+=f1_score(true_labels,pred_labels)
            if epoch % self.print_every == 0:
                print('Training Accuracy : {}'.format(training_acc/len(train_loader)))
            self.__validate(model,model_save_path,validation_data)
            self.training_loss_history.append(training_loss/len(train_loader))
            self.training_accuracy_history.append(training_acc/len(train_loader))
            self.training_f1_score_history.append(training_f1/len(train_loader))

    def __validate(self,model:ELMo,model_save_path,validation_data):
        model.eval()
        model.to(self.device)
        validation_loader = validation_data.get_batches(self.batch_size)
        validation_loss = 0
        validation_acc = 0
        validation_f1 = 0
        threshold = 0.5
        for (sent_data,target) in tqdm(validation_loader):
                sent_data = sent_data.to(self.device)
                target = target.to(self.device)
                output = model(sent_data)
                loss = self.criterion(output, target)
                validation_loss += loss.item()
                true_labels = target.cpu().detach().numpy()
                pred_labels = output.cpu().detach().numpy()
                true_labels = (true_labels >= threshold).astype(np.int64)
                pred_labels = (pred_labels >= threshold).astype(np.int64)
                validation_acc+=accuracy_score(true_labels,pred_labels)
                validation_f1+=f1_score(true_labels,pred_labels)
        if validation_loss < self.lowest_validation_loss:
            self.lowest_validation_loss = validation_loss
            torch.save(model.state_dict(), model_save_path)
        print('Validation Accuracy : {}'.format(validation_acc/len(validation_loader)))

    def plot_loss(self):
        plt.plot(self.training_loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.training_accuracy_history)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve')
        plt.show()

    def plot_f1_score(self):
        plt.plot(self.training_f1_score_history)
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Training F1 Score Curve')
        plt.show()

    