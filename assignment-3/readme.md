
# Assignment - 3

## Info

Most of the parameters are configurable using the `config.json` file. Here are the properties that are configurable.

### Configuration

```json
{
    "cleaning": {
        "min_sentence_length": 10,
        "sentence_sample": true,
        "sentence_sample_size": 80000
    },
    "path": {
        "models": "models",
        "corpus_file": "./data/reviews_Movies_and_TV.json",
        "clean_corpus_file": "corpus_cleaned.txt",
        "data_folder": "data",
        "embeddings": "embeddings",
        "figures": "figures"
    },
    "models": {
        "cbow_neg": {
            "embedding_size": 300,
            "window_size": 7,
            "min_freq": 5,
            "epochs": 25,
            "batch_size": 512,
            "lr": 0.001,
            "neg_sample": 5,
            "print_every": 2,
            "checkpoint_every": 5
        },
        "svd": {
            "embedding_size": 300,
            "window_size": 7,
            "min_freq": 5
        }
    }
}
```

## Execution Instructions

### Cleaning

Clean the corpus file

```bash
python3 main.py clean
```

### Train the model

1. Train the CBOW model with negative sampling

```bash
python3 main.py train cbow_neg
```

2. Train the SVD model

```bash
python3 main.py train svd
```

### Plot the embeddings with T-SNE
```bash
python3 main.py plot embeddings/cbow_neg/cbow_neg_embeddings_300.txt titanic svd
```