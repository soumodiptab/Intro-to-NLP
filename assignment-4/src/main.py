import sys
import json
import os

from dataset import SstData,MultiNliData,ElmoDataset
from utils import load_embeddings
from elmo import ELMo
from sentiment_classifier import SentimentClassifier
from trainer import ElmoTrainer,SentTrainer

def sst_training():
    sst_train = SstData("data/sst.hf","train",80,1)
    sst_validation = SstData("data/sst.hf","validation",80,1,sst_train.vocab)
    sst_test = SstData("data/sst.hf","test",80,1,sst_train.vocab)
    sst_elmo_train = ElmoDataset(sst_train)
    sst_elmo_validation = ElmoDataset(sst_validation)
    sst_elmo_test = ElmoDataset(sst_test)
    glove = load_embeddings(sst_train.vocab,"data/glove.6B.100d.txt",100)
    elmo = ELMo(len(sst_elmo_train.vocab),100,100,80,glove)
    trainer = ElmoTrainer(epochs=20,lr=0.001,batch_size=50,print_every=1,device='cuda')
    trainer.train(elmo,'model/elmo.pt',sst_elmo_train,sst_elmo_validation)
    trainer.plot_loss()
    elmo_embeddings = elmo.get_embeddings(sst_elmo_test)
    elmo_lstm1_param = elmo.lstm1
    elmo_lstm2_param = elmo.lstm2
    sent_classifier = SentimentClassifier(100,100,elmo_embeddings,elmo_lstm1_param,elmo_lstm2_param)
    sent_trainer = SentTrainer(epochs=20,lr=0.001,batch_size=50,print_every=1,device='cuda')
    sent_trainer.train(sent_classifier,'model/sent_classifier.pt',sst_train,sst_validation)
    sent_trainer.plot_loss()
    sent_trainer.plot_accuracy()
    sent_trainer.plot_f1_score()

def mnli_training():
    multi_nli_train = MultiNliData('data/multi_nli.hf','train',200,3,20000)
    multi_nli_validation = MultiNliData('data/multi_nli.hf','validation_matched',200,3,8000,multi_nli_train.get_vocab())
    multi_nli_elmo_train = ElmoDataset(multi_nli_train)
    multi_nli_elmo_validation = ElmoDataset(multi_nli_validation)
    glove = load_embeddings(multi_nli_elmo_train.vocab,"data/glove.6B.100d.txt",100)
    trainer2 = ElmoTrainer(epochs=20,lr=0.001,batch_size=64,print_every=1,device='cuda')
    elmo2 = ELMo(len(multi_nli_elmo_train.vocab),100,100,200,glove)
    trainer2.train(elmo2,'model/elmo2.pt',multi_nli_elmo_train,multi_nli_elmo_validation)
    trainer2.plot_loss()




def train_elmo(dataset_type):
    if dataset_type == 'sst':
        sst_training()
    else:
        mnli_training()


def main():
    print("Starting training")
    dataset_type = sys.argv[1]
    train_elmo(dataset_type)

main()