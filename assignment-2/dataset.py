import torch
from torch.utils.data import Dataset, DataLoader

class POSDataSet(Dataset):
    def __init__(self, connlu_data, vocab, tags, max_seq_len):
        x,y=self.__create_data(connlu_data, vocab, tags, max_seq_len)
        self.vocab = vocab
        self.tags = tags
        self.max_seq_len = max_seq_len
        self.sent = torch.LongTensor(x)
        self.sent_tags = torch.LongTensor(y)
    

    def __create_data(self,sentences,vocab,tags,max_seq_len=50):
        sents_idx=[]
        sent_tags=[]
        for sent in sentences:
            sent_idx=[]
            sent_tag=[]
            for token in sent:
                if (token["form"].lower() in vocab):
                    sent_idx.append(vocab[token["form"].lower()])
                else:
                    sent_idx.append(vocab["<UNK>"])
                sent_tag.append(tags[token["upos"]])
            sents_idx.append(sent_idx)
            sent_tags.append(sent_tag)
        for i in range(len(sents_idx)):
            if len(sents_idx[i]) < max_seq_len:
                sents_idx[i]=sents_idx[i]+[vocab["<PAD>"] for _ in range(max_seq_len - len(sents_idx[i]))]
                sent_tags[i]=sent_tags[i]+[tags["PAD"] for _ in range(max_seq_len - len(sent_tags[i]))]
        return sents_idx,sent_tags
    
    def __getitem__(self, idx):
        return self.sent[idx], self.sent_tags[idx]
    
    def __len__(self):
        return len(self.sent)
    
    def get_dataloader(self, batch_size,shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    

    
