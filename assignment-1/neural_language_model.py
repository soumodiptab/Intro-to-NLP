import random
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential,load_model
import tensorflow.keras.utils as ku 
import numpy as np
import math
import re
from nltk import sent_tokenize
from tokenizer import Tokenizer
import argparse

max_len_allowed=1e2
flag_ulysses = True



def read_from_file(file_name):
    text_lines=[]
    if(flag_ulysses):
        with open(file_name, "r") as f:
            text_lines = f.readlines()
    else:
        with open(file_name, "r") as f:
            text = f.read()
        text_lines = sent_tokenize(text)
    actual=[]
    for text in text_lines:
        if not text.rstrip('\n') == "":
            actual.append(text.rstrip('\n'))
    return actual


def get_sequences_of_tokens(strings):
    op=[]
    counters=dict()
    tk = Tokenizer()
    # strings=lines(corpus)
    i=0
    actual=0
    for line in strings:
        lists=tk.tokenize(line)
        # for word in list:
        #     op.append(word)
        i+=1
        if(len(lists)<=1 or len(lists) > max_len_allowed):
            continue
        actual+=1
        op.append(lists)
    print("Truncated lines:"+str(i)+"\t Actual Lines: "+str(actual))
    return op

def handle_preprocess(corpus):
    dataset = read_from_file(corpus)
    data = get_sequences_of_tokens(dataset)
    # print(data)
    # for line in data:
    #     # line.insert(0, '<s>')
    #     # line.insert(0, '<s>')
    #     line.insert(0, '<s>')
    #     line.append('</s>')
    return data


def model_creation(MODEL_PATH):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Provide model path")
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    if not os.path.exists(MODEL_PATH):
        error('Model file does not exist.')
        model_creation(MODEL_PATH)
        exit(1)