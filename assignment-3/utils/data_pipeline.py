import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import random
class DataPipeline(Dataset):
    def __init__(self, filename,window_size = 4,min_freq=1,vocab=None,neg_words=5):
        self.data = self.read_data(filename)
        self.neg_words = neg_words
        self.window_size = window_size
        if vocab is None:
            self.vocab, self.ind2vocab,self.word_count = self.build_vocab(self.data)
        else:
            self.vocab = vocab
            self.ind2vocab = {v: k for k, v in vocab.items()}
            self.word_count = self.get_word_count(vocab,self.data)
        self.neg_sampling_table = self.__create_neg_sampling_table()
        self.sub_sampling_table = self.__create_sub_sampling_table()

    def get_vocab(self):
        return self.vocab

    def read_data(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                e = line.strip()
                data.append(e.split())
        return data
    
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

    def build_vocab(self, data,min_freq=1):
        word_set = {}
        for line in data:
            for word in line:
                if word not in word_set:
                    word_set[word]=1
                else:
                    word_set[word]+=1
        # sort the vocab
        word_list = sorted(list(word_set))
        word_count = {0: 1}
        vocab_dict = {"<unk>": 0}
        i=1
        for word in word_list:
            if word_set[word] >= min_freq:
                vocab_dict[word] = i
                word_count[i] = word_set[word]
                i+=1
            else:
                word_count[0] += word_set[word]
        ind2word = {v: k for k, v in vocab_dict.items()}
        return vocab_dict, ind2word, word_count

    def total_count(self):
        return sum(self.word_count.values())

    
    def __create_sub_sampling_table(self, threshold=1e-5):
        word_freq = np.array(list(self.word_count.values()))
        word_freq = word_freq / np.sum(word_freq)
        sub_sampling_table = ((np.sqrt(word_freq / threshold) + 1) * (threshold / word_freq))
        return sub_sampling_table
    
    def is_sample_selected(self, idx):
        # return True if the word is selected
        return random.random() < self.sub_sampling_table[idx]
    
    def __create_neg_sampling_table(self, power=0.75, table_size =1e8):
        vocab_size = len(self.vocab)
        word_freq = np.array(list(self.word_count.values())) ** power
        word_freq = word_freq / np.sum(word_freq)
        count = np.round(word_freq * table_size)
        neg_sampling_table = []
        for i in range(vocab_size):
            neg_sampling_table += [i] * int(count[i])
        neg_sampling_table = np.array(neg_sampling_table)
        np.random.shuffle(neg_sampling_table)
        return neg_sampling_table.tolist()
    
    def get_negative_samples(self, target, k):
        delta = random.sample(self.neg_sampling_table, k)
        while target in delta:
            delta = random.sample(self.neg_sampling_table, k)
        return delta   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx]
        if len(words) < self.window_size:
            raise Exception("Sentence length is less than window size")
        data = []
        start = self.window_size // 2
        for i in range(start, len(words) - start):
            target = self.vocab[words[i]]
            if not self.is_sample_selected(target):
                continue
            context = words[i - start: i] + words[i + 1: i + start + 1]
            context = [self.vocab[word] for word in context]
            neg_samples = self.get_negative_samples(target, self.neg_words)
            data.append((target, context, neg_samples))
        return data
    
    def __collate_fn(self,batches):
        target = []
        context = []
        neg_samples = []
        for sentence in batches:
            for t,c,n in sentence:
                target.append(t)
                context.append(c)
                neg_samples.append(n)
        return torch.LongTensor(target),torch.LongTensor(context),torch.LongTensor(neg_samples)

    def get_batches(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False,collate_fn=self.__collate_fn ,drop_last=True)
