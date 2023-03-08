# Assignment -2 NLP
## Neural POS Tagger

## Files Structure:

```bash
.
├── config.json
├── dataset.py
├── model.py
├── pos_tagger.py
├── readme.md
├── tags.json
└── utils.py

```

## Config File:
The pos tagger is configurable with the following structure:
```json
{
    "DATA_FOLDER": "data/UD_English-Atis/",
    "TRAIN_FILE": "en_atis-ud-train.conllu",
    "TEST_FILE": "en_atis-ud-test.conllu",
    "DEV_FILE": "en_atis-ud-dev.conllu",
    "WORD_EMBEDDINGS_FILE": "data/glove.6B.100d.txt",
    "BATCH_SIZE": 32,
    "SEED":159,
    "EPOCHS": 100,
    "EMBEDDING_DIM": 100,
    "HIDDEN_DIM": 128,
    "DROPOUT": 0.8,
    "N_LAYERS": 2,
    "LEARNING_RATE": 0.0005,
    "MODEL_PATH": "pos_tagger2.pt",
    "MAX_SEQ_LENGTH": 50,
    "VOCAB_SIZE": 50000,
    "BIDIRECTIONAL": true

}
```

## Instructions :

1. Running the file in train mode
```bash
python3 pos_tagger.py train
```
2. Running the file in evaluation mode
```bash
python3 pos_tagger.py eval
```
3. Running the file to test sentences
```bash
python3 pos_tagger.py
```
### Embeddings Link :
[Google Drive](https://drive.google.com/drive/folders/1MvXBOP0sJpygn6wLnGuSlsqQglv2nUs6?usp=sharing)


### Model link:
[Google Drive](https://drive.google.com/drive/folders/1QKUU0w7etRXGvFEKYSeKn5LTMVLIDlhm?usp=share_link)