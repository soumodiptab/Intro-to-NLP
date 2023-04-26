
from  tqdm import tqdm
from collections import Counter
import torch
import numpy as np
from datasets import load_dataset,load_from_disk
from torch.utils.data import Dataset,DataLoader
from datacleaner import DataCleaner

class DataPipeline(Dataset):
    def __init__(self, filename,type,max_seq_len=50,min_freq=3,vocab=None):
        self.read_data(filename,type)
        self.max_seq_len = max_seq_len
        if vocab is None:
            self.vocab, self.ind2vocab,self.word_count = self.build_vocab(self.data,min_freq)
        else:
            self.vocab = vocab
            self.ind2vocab = {v: k for k, v in vocab.items()}
            # self.word_count = self.get_word_count(vocab,self.data)
        self.ind2vocab = {ind: word for word, ind in self.vocab.items()}
        

    def get_vocab(self):
        return self.vocab
    def word_to_ind(self,word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return 1
    def ind_to_word(self,ind):
        return self.ind2vocab[ind]
    
    def read_data(self,filename,type):
        pass
    
    def get_word_count(self,vocab,data):
        word_count = {0: 0}
        for line in data:
            for word in line:
                if word in vocab:
                    word_count[vocab[word]] += 1
                else:
                    word_count[0] += 1
        return word_count
    
    def most_common(self,n):
        counter = Counter(self.word_count)
        common = counter.most_common(n)
        ind_freq = dict(common)
        # convert to word frequency
        word_freq = {}
        for ind in ind_freq:
            word_freq[self.ind2vocab[ind]] = ind_freq[ind]
        return word_freq
    
    @staticmethod
    def build_vocab(data,min_freq):
        word_set = {}
        print('Building vocab:')
        for line in tqdm(data):
            for word in line:
                if word not in word_set:
                    word_set[word]=1
                else:
                    word_set[word]+=1
        # sort the vocab
        word_list = sorted(list(word_set))
        word_count = {0: 0, 1: 0}
        vocab_dict = {"<pad>":0,"<unk>": 1}
        i=2
        for word in tqdm(word_list):
            if word_set[word] >= min_freq:
                vocab_dict[word] = i
                word_count[i] = word_set[word]
                i+=1
            else:
                word_count[0] += word_set[word]
        ind2word = {v: k for k, v in vocab_dict.items()}
        print('Vocab size: {}'.format(len(vocab_dict)))
        return vocab_dict, ind2word, word_count

    def total_count(self):
        return sum(self.word_count.values())

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
    
    def get_batches(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False,drop_last=True)
    

class SstData(DataPipeline):
    def read_data(self, filename, type):
        datacleaner = DataCleaner()
        data =load_from_disk(filename)
        processed_data = []
        target = []
        for line in tqdm(data[type]):
            processed_data.append(datacleaner.process(line['sentence']).split(" "))
            target.append(line['label'])
        self.data=processed_data
        self.target=target
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sent = self.data[idx]
        label = self.target[idx]
        # paddding the sentences to create sequences of same length
        if len(sent) < self.max_seq_len:
            sent=[self.word_to_ind(token) for token in sent]+[self.word_to_ind("<pad>") for _ in range(self.max_seq_len - len(sent))]
        return torch.LongTensor(sent),torch.Tensor([label])
    
class MultiNliData(DataPipeline):
    def __init__(self, filename,type,max_seq_len=50,min_freq=3,sentence_limit=50000,vocab=None):
        self.sentence_limit = sentence_limit
        super().__init__(filename,type,max_seq_len,min_freq,vocab)
    def read_data(self, filename, type):
        datacleaner = DataCleaner()
        data =load_from_disk(filename)
        processed_data = []
        premise = []
        hypothesis = []
        target = []
        counter = 0
        for line in tqdm(data[type]):
            if counter > self.sentence_limit:
                break
            counter += 1
            p = datacleaner.process(line['premise']).split(" ")
            h = datacleaner.process(line['hypothesis']).split(" ")
            processed_data.append(p)
            processed_data.append(h)
            premise.append(p)
            hypothesis.append(h)
            target.append(line['label'])
        self.data = processed_data
        self.target = target
        self.premise = premise
        self.hypothesis = hypothesis
    def __len__(self):
        return len(self.premise)
    def __getitem__(self, idx):
        premise = self.premise[idx]
        hypothesis = self.hypothesis[idx]
        label = self.target[idx]
        # paddding the sentences to create sequences of same length
        if len(premise) < self.max_seq_len:
            premise=[self.word_to_ind(token) for token in premise]+[self.word_to_ind("<pad>") for _ in range(self.max_seq_len - len(premise))]
        if len(hypothesis) < self.max_seq_len:
            hypothesis=[self.word_to_ind(token) for token in hypothesis]+[self.word_to_ind("<pad>") for _ in range(self.max_seq_len - len(hypothesis))]
        return torch.LongTensor(premise),torch.LongTensor(hypothesis),torch.Tensor([label])

class ElmoDataset(Dataset):
    def __init__(self,dataset :DataPipeline):
        self.data = dataset.data
        self.max_seq_len = dataset.max_seq_len
        self.vocab = dataset.vocab
        self.ind2vocab = dataset.ind2vocab
        self.word_to_ind = dataset.word_to_ind
        self.ind_to_word = dataset.ind_to_word
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sent = self.data[idx]
        # paddding the sentences to create sequences of same length
        if len(sent) < self.max_seq_len:
            sent=[self.word_to_ind(token) for token in sent]+[self.word_to_ind("<pad>") for _ in range(self.max_seq_len - len(sent))]
        forward_data = sent[1:]
        backward_data = sent[:-1]
        return torch.LongTensor(forward_data),torch.LongTensor(backward_data)
    def get_batches(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False,drop_last=True)