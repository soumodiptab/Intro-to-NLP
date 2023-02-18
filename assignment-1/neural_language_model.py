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
import os
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
        lists=tk.tokenize2(line)
        # for word in list:
        #     op.append(word)
        i+=1
        if(len(lists)<=1 or len(lists) > max_len_allowed):
            continue
        actual+=1
        op.append(lists)
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

def pad_sequences(sequences, maxlen, pad_index=0):
    # Pad each sequence with the given value
    padded_sequences = []
    for seq in sequences:
        mult_seq =(maxlen - len(seq))
        padded_seq = [pad_index] * mult_seq  + seq
        padded_sequences.append(padded_seq)    
    return np.array(padded_sequences)


class Model:
    def __init__(self,predictors, label, input_len, total_words,X_val,Y_val,path):
        model = Sequential()
        model.add(Embedding(total_words, 12, input_length=input_len))
        model.add(LSTM(132))
        model.add(Dropout(0.35))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        stop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, verbose = 1, mode = 'auto')
        save = ModelCheckpoint(path, monitor = 'val_loss', verbose = 1, save_best_only = True)
        callbacks = [stop, save]
        print(model.summary())
        final=model.fit(predictors, label, validation_data=(X_val,Y_val),epochs=80, batch_size=128, verbose=1,callbacks = callbacks)
        return final
    
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
  min_count = 2
  unknown_token = '<unk>'
  word2index = {unknown_token: 1}
  index2word = [unknown_token]
  id=2
  for word, count in counter.items():
      if count >= min_count:
          index2word.append(word)
          word2index[word] = id
          id+=1
  num_classes = len(word2index)
  return word2index,index2word,num_classes


def handle_preprocess_sent(sent):
    line = get_sequences_of_tokens([sent])
    # line[0].insert(0, '<s>')
    # line[0].insert(0, '<s>')
    # line[0].insert(0, '<s>')
    # line[0].append('</s>')
    return line[0]

def find_prob(X_data,Y_data,Y_pred):
    ret_prob = 1
    for lines in X_data:
        for i in range(len(lines)-1):
            ret_prob*= Y_pred[i][np.argmax(Y_data[i])]
    return ret_prob

def build_input(tokenized_data,word2index,unknown_token,num_class,mx):
  input_sequences=[]
  mxx=mx
  for token_list in tokenized_data:
    n_list=[]
    for word in token_list:
        if word not in word2index:
            word=unknown_token
        seq_num=word2index[word]
        n_list.append(seq_num)
    mxx=max(mxx,len(n_list))
    for i in range(1, len(n_list)):
        n_gram_sequence = n_list[:i+1]
        input_sequences.append(n_gram_sequence)
  input_sequences=pad_sequences(input_sequences,mxx)

  predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
  label = ku.to_categorical(label, num_classes=num_class)
#   print(mxx)
  return predictors,label

def model_creation(MODEL_PATH):
    pass

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("model_path", help="Provide model path")
    # parser.add_argument("corpus_path", help="Provide corpus path")
    # args = parser.parse_args()
    #MODEL_PATH = args.model_path
    #CORPUS_PATH = args.corpus_path
    CORPUS_PATH = "./corpora/Pride and Prejudice - Jane Austen.txt"
    MODEL_PATH ="./models/lstm_weights.hdf5"
    if not os.path.exists(MODEL_PATH):
        error('Model file does not exist.')
        model_creation(MODEL_PATH)
        exit(1)
    else:
        info('Model file exists.')
        model = load_model(MODEL_PATH)
        info('Model loaded.')
        test_text = input('Input Sentence: ')
        dataset=handle_preprocess(CORPUS_PATH)
        cnter,total,mx=gen_freq_table(dataset)
        word2id,id2word,num_classes=build_vocab(cnter)
        num_classes+=1
        p = input("input sentence: ")
        token_sen=handle_preprocess_sent(test_text)
        sen_x,sen_y=build_input(token_sen,word2id,'<unk>',num_classes,mx)
        # print(sen_x.shape)
        # print(sen_y.shape)
        y_pred_sen=model.predict(sen_x)
        # print(y_pred_sen.shape)
        vals = find_prob(token_sen, sen_y, y_pred_sen)
        print(vals)
