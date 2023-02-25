import random 
import numpy as np
import math
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from nltk import sent_tokenize
from tokenizer import Tokenizer
import tensorflow.keras.utils as kutils
import argparse
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import os
from tensorflow.keras.models import Sequential,load_model
import re
CORPUS_NAME = ""
max_len_allowed=1e2
flag_ulysses = False
def decorate():
    print("=================================================================================================")

def error(msg):
    decorate()
    print("[ERROR] : " + msg)



def info(msg):
    decorate()
    print("[INFO] : " + msg)

def read_from_file(file_name):
    text_lines=[]
    if(CORPUS_NAME == "Ulysses - James Joyce"):
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


def token_sequencer(text_lines):
    tk = Tokenizer()
    res=[]
    i=0
    actual=0
    for text_line in text_lines:
        lists=tk.tokenize2(text_line)
        i+=1
        if(len(lists)<=1 or len(lists) > max_len_allowed):
            continue
        actual+=1
        res.append(lists)
    return res

def preprocess_handler(corpus):
    dataset = read_from_file(corpus)
    data = token_sequencer(dataset)
    return data

def parse_input(tokenized_data,word_to_index,token_unk,num_class,maxi):
  local_maxi=maxi
  input_tokens=[]
  for token_list in tokenized_data:
    index_list=[]
    for word in token_list:
        try :
            _=word_to_index[word]
        except KeyError:
            word=token_unk
        seq_num=word_to_index[word]
        index_list.append(seq_num)
    local_maxi=max(len(index_list),local_maxi)
    limit =len(index_list)
    for n in range(1, limit):
        n_gram_sequence = index_list[:n+1]
        input_tokens.append(n_gram_sequence)
  input_tokens=pad_sequences(input_tokens,local_maxi)
  predicted_values = input_tokens[:,:-1]
  label_values=input_tokens[:,-1]
  label_values = kutils.to_categorical(label_values, num_classes=num_class)
  return label_values,predicted_values

def pad_sequences(sequences, maxlen, pad_index=0):
    # Pad each sequence with the given value
    padded_sequences = []
    for seq in sequences:
        mult_seq =abs(maxlen - len(seq))
        padded_seq = [pad_index] * mult_seq  + seq
        padded_sequences.append(padded_seq)    
    return np.array(padded_sequences)


    
def gen_freq_table(text):
  t=0
  max_len=0
  total_len=0
  freq_table={}
  for line in text:
    for word in line:
        if not word in freq_table:
            freq_table[word]=1
        else:
            freq_table[word]+=1
        total_len=total_len+len(line)
        if len(line)> max_len:
         max_len=len(line)      
  return freq_table,total_len,max_len 

def build_vocab(counter):
  unknown_token = '<unkown>'
  index_to_word = [unknown_token]
  word_to_index = {unknown_token: 1}
  min_count = 3
  idx=2
  for word, count in counter.items():
    if min_count <= count:
        word_to_index[word] = idx
        idx=idx+1
        index_to_word.append(word)
  return word_to_index,index_to_word,len(word_to_index)+1


def sentence_preprocessor(sent):
    return token_sequencer([sent])[0]

def get_probability(x_data,y_data,pred_values_y):
    prob_value = 1
    for lines in x_data:
        limit =len(lines)-1
        for i in range(limit):
            idx =np.argmax(y_data[i])
            prob_value= pred_values_y[i][idx]*prob_value
    return prob_value

def model_creation(MODEL_PATH):
    pass

def model_get_predicted_value(model,data):
    return model.predict(data)


def model_pred(MODEL_PATH,CORPUS_PATH):
    model = load_model(MODEL_PATH)
    info('Model loaded.')
    test_text = input('Input Sentence: ')
    dataset=preprocess_handler(CORPUS_PATH)
    CORPUS_NAME = CORPUS_PATH.split("/")[-1].split(".")[0]
    central_dict,_,maxi=gen_freq_table(dataset)
    word_to_index,_,num_of_classes=build_vocab(central_dict)
    token_sen=sentence_preprocessor(test_text)
    sentence_y,sentence_x=parse_input(token_sen,word_to_index,'<unkown>',num_of_classes,maxi)
    model_pred_values = model_get_predicted_value(model,sentence_x)
    vals = get_probability(token_sen, sentence_y, model_pred_values)
    print(vals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Provide model path")
    parser.add_argument("corpus_path", help="Provide corpus path")
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    CORPUS_PATH = args.corpus_path
    #CORPUS_PATH = "./corpora/Pride and Prejudice - Jane Austen.txt"
    #MODEL_PATH ="./models/lstm_weights.hdf5"
    if not os.path.exists(MODEL_PATH):
        error('Model file does not exist.')
        model_creation(MODEL_PATH)
        exit(1)
    else:
        info('Model file exists.')
        model_pred(MODEL_PATH,CORPUS_PATH)
        
