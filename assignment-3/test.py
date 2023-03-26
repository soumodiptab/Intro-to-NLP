import sys
import json
from data_cleaner import clean_corpus
from data_pipeline import DataPipeline
from svd import SVD_W2V
from cbow import CBOW_NEG
from plotter import plot_top10_words
import os
import torch
# arguments : clean, train, plot
config =json.loads(open('config.json','r').read())
path_config = config['path']
models_config = config['models']
cleaner_config = config['cleaning']


DATA_FOLDER = path_config['data_folder']
EMBEDDINGS_FOLDER = path_config['embeddings']
MODEL_FOLDER = path_config['models']
FIGURES_FOLDER = path_config['figures']

def cbow_trainer():
    corpus_file = os.path.join(DATA_FOLDER,path_config['clean_corpus_file'])
    print('Loading data from {}'.format(corpus_file))
    dataset = DataPipeline(corpus_file,
                           min_freq=models_config['cbow_neg']['min_freq'],
                           window_size=models_config['cbow_neg']['window_size'],
                           neg_words=models_config['cbow_neg']['neg_sample'])
    
    # model = CBOW_NEG(vocab_size=len(dataset.vocab),
    #                  embedding_size=models_config['cbow_neg']['embedding_size'],
    #                  model_path=MODEL_FOLDER,
    #                  embedding_path=os.path.join(EMBEDDINGS_FOLDER,'cbow_neg'))
    # model.load_model(os.path.join(MODEL_FOLDER,'cbow_neg_lr_0.001_e_300.pth'))
    model = torch.load(os.path.join(MODEL_FOLDER,'cbow_neg_lr_0.001_e_300.pth'))
    print(' Model loaded')
cbow_trainer()