import sys
import json
from data_cleaner import clean_corpus
from data_pipeline import DataPipeline
from svd import SVD_W2V
from cbow import CBOW_NEG
from plotter import plot_top10_words
import os
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
    
    model = CBOW_NEG(vocab_size=len(dataset.vocab),
                     embedding_size=models_config['cbow_neg']['embedding_size'],
                     model_path=MODEL_FOLDER,
                     embedding_path=os.path.join(EMBEDDINGS_FOLDER,'cbow_neg'))
    model.trainer(dataset,
                  batch_size=models_config['cbow_neg']['batch_size'],
                  epochs=models_config['cbow_neg']['epochs'],
                  lr=models_config['cbow_neg']['lr'],
                  print_every=models_config['cbow_neg']['print_every'],
                  checkpoint_every=models_config['cbow_neg']['checkpoint_every'])


def svd_trainer():
    corpus_file = os.path.join(DATA_FOLDER,path_config['clean_corpus_file'])
    print('Loading data from {} ...'.format(corpus_file))
    data = DataPipeline.read_data(corpus_file)
    vocab, _, _ = DataPipeline.build_vocab(data,min_freq=models_config['svd']['min_freq'])
    print('Training SVD model with embedding size : {} Window : {}'.format(
        models_config['svd']['embedding_size'],
        models_config['svd']['window_size']))
    model = SVD_W2V(vocab,models_config['svd']['window_size'],models_config['svd']['embedding_size'])
    model.train(data)
    embeddings_file = os.path.join(EMBEDDINGS_FOLDER,"svd",'svd_embeddings_{}.txt'.format(models_config['svd']['embedding_size']))
    model.save_embeddings(embeddings_file)
    print('Embeddings saved at {}'.format(embeddings_file))

def trainer(model_name):
    if model_name == 'cbow_neg':
        cbow_trainer()
    elif model_name == 'svd':
        svd_trainer()
    else:
        print('Please provide proper arguments: clean, train [cbow_neg,svd], plot [cbow_neg,svd] [word]')
        exit(1)


def plotter(embeddings_path, word,FOLDER):
    if not os.path.exists(embeddings_path):
        print('Embeddings file not found at {}'.format(embeddings_path))
        exit(1)
    file_save_path = os.path.join(FIGURES_FOLDER,FOLDER)
    if not os.path.exists(file_save_path):
        os.mkdir(file_save_path)
    plot_top10_words(embeddings_path,word,file_save_path)

def cleaner():
    clean_corpus(
        path_config['corpus_file'],
        os.path.join(DATA_FOLDER,path_config['clean_corpus_file']),
        cleaner_config['sentence_sample'],
        cleaner_config['sentence_sample_size'],
        cleaner_config['min_sentence_length']
    )


if __name__ == '__main__':
    if not os.path.exists(EMBEDDINGS_FOLDER):
        os.mkdir(EMBEDDINGS_FOLDER)
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    if not os.path.exists(FIGURES_FOLDER):
        os.mkdir(FIGURES_FOLDER)
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    if len(sys.argv) < 2 :
        print('Please provide proper arguments: clean, train [cbow_neg,svd], plot [cbow_neg,svd] [word]')
        exit(1)
    if sys.argv[1] == 'clean':
        cleaner()
    elif sys.argv[1] == 'train' and len(sys.argv) == 3:
        trainer(sys.argv[2])
    elif sys.argv[1] == 'plot' and len(sys.argv) == 5:
        plotter(sys.argv[2], sys.argv[3],sys.argv[4])
    else:
        print('Please provide proper arguments: clean, train [cbow_neg,svd], plot [embeddings_path] [word] [folder]')
        exit(1)