import tokenizer as tk
import argparse
import os
import pickle
import numpy as np
import logging
import time
ROLL_NO = "2021201086"


def error(msg):
    print("[Error] : " + msg)


def info(msg):
    print("[Info] : " + msg)


class Ngram:
    def __init__(self):
        pass

    def get_unigram(self, text_lines):
        unigram = []
        for text in text_lines:
            unigram += text.split(' ')
        return unigram

    def __get_ngram(self, text_line, n):
        ngram = []
        tokens = text_line.split(' ')
        if len(tokens) == 0:
            return None
        tokens = ["<BEGIN>" for _ in range(n-2)] + tokens
        for i in range(len(text_line)-n+1):
            ngram.append(" ".join(tokens[i:i+n]))
        return ngram

    def generate_ngram(self, text_lines,n):
        for text in text_lines:
            ngram = self.__get_ngram(text, n)
            if ngram is not None:
                yield ngram


def test_train_split(data, test_distribution=0.2, flag=False):
    test_size = 1000
    if flag:
        test_size = n*test_distribution
    np.random.seed(time.time())
    n = len(data)
    idx_list = np.random.choice(n, int(test_size), replace=False)
    train_data = []
    test_data = []
    for i in range(len(data)):
        if i in idx_list:
            test_data.append(data[i])
        else:
            train_data.append(data[i])
    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="Ngram value")
    parser.add_argument("smoothing", help="Smoothing technique to be used")
    parser.add_argument("corpus", help="Path to the corpus file")
    args = parser.parse_args()
    smoothing_technique = args.smoothing
    CORPUS_PATH = args.corpus
    N = int(args.n)
    corpus_name = CORPUS_PATH.split("/")[-1].split(".")[0]
    if not os.path.exists(CORPUS_PATH):
        error('Corpus file does not exist.')
        exit(1)
    CLEAN_CORPORA_PATH = os.path.join("clean_corpora", corpus_name+".txt")
    clean_text_lines = []
    if not os.path.exists(CLEAN_CORPORA_PATH):
        tokenizer = tk.Tokenizer()
        text_lines = tk.read_from_file(CORPUS_PATH)
        clean_text_lines = []
        for text in text_lines:
            clean_text_line = tokenizer.tokenize(text)
            if clean_text_line.strip() != "":
                clean_text_lines.append(clean_text_line)
        tk.save_to_file(CLEAN_CORPORA_PATH, clean_text_lines)
    else:
        clean_text_lines = tk.read_from_file(CLEAN_CORPORA_PATH)
    train, test = test_train_split(clean_text_lines)
    if not os.path.exists():
