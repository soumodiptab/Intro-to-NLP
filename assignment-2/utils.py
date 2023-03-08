import numpy as np
import torch
from conllu import parse

def set_seed(seed):
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)

def get_conllu_data(filename):
    with open (filename, "r", encoding="utf-8") as f:
        data = f.read()
    sentences = parse(data)
    return sentences

def accuracy(true, pred):
  true = np.array(true)
  pred = np.array(pred)

  num_correct = sum(true == pred)
  num_total = len(true)
  return num_correct / num_total

def get_uniq_words(sentences):
    words = set()
    for sentence in sentences:
        for token in sentence:
            words.add(token["form"].lower())
    return words

def read_embeddings(filename, vocab_size=10000):
    with open(filename, encoding="utf-8") as file:
        word_embedding_dim = len(file.readline().split(" ")) - 1
    vocab = {}
    embeddings = np.zeros((vocab_size, word_embedding_dim))
    with open(filename, encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if idx + 2 >= vocab_size:
                break
            cols = line.rstrip().split(" ")
            val = np.array(cols[1:])
            word = cols[0]
            embeddings[idx + 2] = val
            vocab[word] = idx + 2
    vocab["<UNK>"]=1
    vocab["<PAD>"]=0
    return torch.FloatTensor(embeddings), vocab

def build_vocab(sentences):
    data =[]
    word_set = set()
    vocab_dict={"<PAD>":0,"<UNK>":1}
    for sent in sentences:
        for token in sent:
            word_set.add(token["form"])
    word_list = sorted(list(word_set))
    for i,word in enumerate(word_list):
        vocab_dict[word]=i+2
    return vocab_dict

def create_tags(sentences):
    tag_set=set()
    for sent in sentences:
        for token in sent:
            tag_set.add(token["upos"])
    tags=sorted(list(tag_set))
    tag_dict={"PAD":0}
    #tag_dict ={}
    for i,tag in enumerate(tags):
        tag_dict[tag]=i+1
    return tag_dict


def sent_to_vector(vocab,sentence,max_seq_len=50):
    tokens = sentence.split(" ")
    sent_idx=[]
    for token in tokens:
        if (token.lower() in vocab):
            sent_idx.append(vocab[token.lower()])
        else:
            sent_idx.append(vocab["<UNK>"])
    for i in range(len(sent_idx)):
        if len(sent_idx) < max_seq_len:
            sent_idx=sent_idx+[vocab["<PAD>"] for _ in range(max_seq_len - len(sent_idx))]
    return sent_idx
