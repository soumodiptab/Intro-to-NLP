import sys
import os
from utils import get_conllu_data,read_embeddings,create_tags,sent_to_vector,set_seed
import json
from dataset import POSDataSet
from model import POSTagger,ImprovedPOSTagger
from sklearn.metrics import classification_report
config_file = "config.json"
CONFIG = json.load(open(config_file, "r"))

def tag_analysis(data_loader,model,tags):
    inv_map = {v: k for k, v in tags.items()}
    acc,true_labels,pred_labels=model.evaluate(data_loader)
    print(classification_report(true_labels,pred_labels,labels=list(tags.values()),target_names=list(tags.keys())))

def run_tagger(MODE):
    if MODE=="train":
        train_sentences=get_conllu_data(CONFIG["DATA_FOLDER"]+CONFIG["TRAIN_FILE"])
        dev_sentences=get_conllu_data(CONFIG["DATA_FOLDER"]+CONFIG["DEV_FILE"])
        embeddings,vocab=read_embeddings(CONFIG["WORD_EMBEDDINGS_FILE"],CONFIG["VOCAB_SIZE"])
        tags=create_tags(train_sentences+dev_sentences)
        print('=====================================================================================================')
        print('Created Tags')
        json.dump(tags,open('tags.json','w'))
        print('=====================================================================================================')
        print('Saved Tags to json file')
        train_dataset = POSDataSet(train_sentences, vocab, tags, CONFIG["MAX_SEQ_LENGTH"])
        print('=====================================================================================================')
        print('Loaded Training Data')
        print('=====================================================================================================')
        dev_dataset = POSDataSet(dev_sentences, vocab, tags, CONFIG["MAX_SEQ_LENGTH"])
        print('Loaded Dev Data')
        print('=====================================================================================================')
        set_seed(CONFIG["SEED"])
        model = ImprovedPOSTagger(
            CONFIG['MAX_SEQ_LENGTH'],
            embeddings,
            CONFIG['HIDDEN_DIM'],
            CONFIG['N_LAYERS'],
            len(tags),
            CONFIG['DROPOUT'],
            CONFIG['BIDIRECTIONAL'])
        model.summary()
        model.run_training(
            train_dataset.get_dataloader(CONFIG["BATCH_SIZE"]), 
            dev_dataset.get_dataloader(CONFIG["BATCH_SIZE"]), 
            CONFIG["EPOCHS"], 
            CONFIG["LEARNING_RATE"])
        model.save(CONFIG["MODEL_PATH"])
        print('=====================================================================================================')
        print(' Model saved to '+CONFIG["MODEL_PATH"]+'')
        print('=====================================================================================================')

    elif MODE=="eval":
        test_sentences=get_conllu_data(CONFIG["DATA_FOLDER"]+CONFIG["TEST_FILE"])
        embeddings,vocab=read_embeddings(CONFIG["WORD_EMBEDDINGS_FILE"],CONFIG["VOCAB_SIZE"])
        tag_dict=json.load(open('tags.json','r'))
        model = ImprovedPOSTagger(
            CONFIG['MAX_SEQ_LENGTH'],
            embeddings,
            CONFIG['HIDDEN_DIM'],
            CONFIG['N_LAYERS'],
            len(tag_dict),
            CONFIG['DROPOUT'],
            CONFIG['BIDIRECTIONAL'],
            'cpu')
        model.summary()
        model.load(CONFIG["MODEL_PATH"])
        print('=====================================================================================================')
        print(' Model loaded from '+CONFIG["MODEL_PATH"]+'')
        print('=====================================================================================================')
        test_dataset = POSDataSet(test_sentences, vocab, tag_dict, CONFIG["MAX_SEQ_LENGTH"])
        acc,true,pred=model.evaluate(test_dataset.get_dataloader(CONFIG["BATCH_SIZE"]))
        print('=====================================================================================================')
        print("Test Accuracy: "+str(acc))
        print('=====================================================================================================')
        print('Tag Analysis:')
        tag_analysis(test_dataset.get_dataloader(CONFIG["BATCH_SIZE"]),model,tag_dict)
        print('=====================================================================================================')

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
