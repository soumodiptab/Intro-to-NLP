# from utils.data_pipeline import DataPipeline
# from models.cbow import CBOWNEG

# dataset = DataPipeline('data/processed_data/corpus_cleaned.txt')
# print('Dataset Loaded')
# model = CBOWNEG(len(dataset.vocab),200)
# print('Model Training :')
# model.trainer(dataset,batch_size=128,epochs=10,lr=0.001,print_every=1,checkpoint_every=2)
import sys
import json
from data_cleaner import clean_corpus
from data_pipeline import DataPipeline
import os
# arguments : clean, train, plot
config =json.loads(open('config.json','r').read())
path_config = config['path']
models_config = config['models']
cleaner_config = config['cleaning']


DATA_FOLDER = path_config['data_folder']
MODEL_FOLDER = path_config['models']
FIGURES_FOLDER = path_config['figures']

def cbow_trainer():
    pass

def svd_trainer():
    pass

def trainer():
    pass


def plotter(model, word):
    pass

def cleaner():
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    clean_corpus(
        path_config['corpus_file'],
        os.path.join(DATA_FOLDER,path_config['clean_corpus_file']),
        cleaner_config['sentence_sample'],
        cleaner_config['sentence_sample_size'],
        cleaner_config['min_sentence_length']
    )


if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print('Please provide proper arguments: clean, train [cbow_neg,svd], plot [cbow_neg,svd] [word]')
        exit(1)
    if sys.argv[1] == 'clean':
        cleaner()
    elif sys.argv[1] == 'train' and len(sys.argv) == 3:
        trainer(sys.argv[2])
    elif sys.argv[1] == 'plot' and len(sys.argv) == 4:
        plotter(sys.argv[2], sys.argv[3])
    else:
        print('Please provide proper arguments: clean, train [cbow_neg,svd], plot [cbow_neg,svd] [word]')
        exit(1)