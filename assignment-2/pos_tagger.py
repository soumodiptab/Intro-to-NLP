from conllu import parse,parse_incr
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from tqdm import tqdm
import sys
from utils import get_conllu_data,read_embeddings,create_tags,sent_to_vector
import json
from model import POSTagger,ImprovedPOSTagger
config_file = "config.json"
CONFIG = json.load(open(config_file, "r"))


def run_tagger(MODE):
    train_sentences=get_conllu_data("data/UD_English-Atis/en_atis-ud-train.conllu")
    dev_sentences=get_conllu_data("data/UD_English-Atis/en_atis-ud-dev.conllu")
    test_sentences=get_conllu_data("data/UD_English-Atis/en_atis-ud-test.conllu")
    if MODE=="train":
        pass
    elif MODE=="eval":
        pass
    else:
        print("Invalid mode selected. Please select from train or eval")
        return

def predictor(model : POSTagger,sentence,vocab,tags,max_seq_len):
    tokens = sentence.split(" ")
    inv_map = {v: k for k, v in tags.items()}
    vec = [sent_to_vector(vocab,sentence,max_seq_len)]
    predictions = model.predict(vec)
    for i in range(len(tokens)):
        print (tokens[i]+"\t"+inv_map[predictions[0][i]])

if __name__ == '__main__':
    if len(sys.argv) == 2 :
        MODE = sys.argv[1]
        run_tagger(MODE)
    else:
        #train_sentences = get_conllu_data("data/UD_English-Atis/en_atis-ud-train.conllu")
        #dev_sentences = get_conllu_data("data/UD_English-Atis/en_atis-ud-dev.conllu")
        embeddings,vocab=read_embeddings("data/glove.6B.100d.txt",CONFIG["VOCAB_SIZE"])
        tag_dict=json.load(open('tags.json','r'))
        tagger = ImprovedPOSTagger(
            CONFIG['MAX_SEQ_LENGTH'],
            embeddings,
            CONFIG['HIDDEN_DIM'],
            CONFIG['N_LAYERS'],
            len(tag_dict),
            CONFIG['DROPOUT'],
            CONFIG['BIDIRECTIONAL'],
            'cpu')
        tagger.load(CONFIG['MODEL_PATH'])
        sentence = input()
        predictor(tagger,sentence,vocab,tag_dict,CONFIG["MAX_SEQ_LENGTH"])
