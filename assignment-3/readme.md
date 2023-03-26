
# Assignment - 3

## Info

Most of the parameters are configurable using the `config.json` file. Here are the properties that are configurable.

### Configuration

```json
{
    "cleaning": {
        "min_sentence_length": 10,
        "sentence_sample": true,
        "sentence_sample_size": 100000
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
            "embedding_size": 400,
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
Format
```
python3 main.py train <model_name>
```
1. Train the CBOW model with negative sampling

```bash
python3 main.py train cbow_neg
```

2. Train the SVD model

```bash
python3 main.py train svd
```

### Load the model and save embeddings
Format
```bash
python3 main.py load <model> <save_path>
```

Example
```bash
python3 main.py load cbow_neg cbow_neg_lr_0.001_e_300.pth
```

### Plot the embeddings with T-SNE
Format :
```bash
python3 main.py plot <embeddding_file_path> <word> <sub_folder_to_save>
```
Example :
```bash
python3 main.py plot embeddings/cbow_neg/cbow_neg_embeddings_300.txt titanic svd
```


## Saved Emdeddings location :
[Download From Here ](https://drive.google.com/drive/folders/1MvXBOP0sJpygn6wLnGuSlsqQglv2nUs6?usp=sharing)

## Clean corpus file location :
Currently sampled on `100000` sentences.

[Download From Here](https://drive.google.com/drive/folders/1Sr5NikUjJOu9x3vm5wvUNtAIa7cG_iVH?usp=sharing)