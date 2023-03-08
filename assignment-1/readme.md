# readme

Folder Structure:
```bash
.
├── 2021201086_Report.pdf
├── corpora
│   ├── Pride and Prejudice - Jane Austen.txt
│   ├── Ulysses - James Joyce.txt
│   └── test1.txt
├── language_model.py
├── models
├── neural_language_model.py
├── readme.md
├── scores
│   ├── 2021201086_LM1_test-perplexity.txt
│   ├── 2021201086_LM1_train-perplexity.txt
│   ├── 2021201086_LM2_test-perplexity.txt
│   ├── 2021201086_LM2_train-perplexity.txt
│   ├── 2021201086_LM3_test-perplexity.txt
│   ├── 2021201086_LM3_train-perplexity.txt
│   ├── 2021201086_LM4_test-perplexity.txt
│   ├── 2021201086_LM4_train-perplexity.txt
│   ├── 2021201086_LM5_test-perplexity.txt
│   ├── 2021201086_LM5_train-perplexity.txt
│   ├── 2021201086_LM6_test-perplexity.txt
│   └── 2021201086_LM6_train-perplexity.txt
└── tokenizer.py
```
### instructions to run file

#### Q1 : Tokenizer

Standalone file - imported in t

#### Q2 : Language model

```bash
python3 language_model.py 4 k "./corpora/Pride and Prejudice - Jane Austen.txt"
```

Run as given in the assignment pdf

#### Q3 : neural_language_model

```bash

python3 neural_language_model.py <model_path> <corpus_path>
```

1. The first param is model.
2. The second param is corpus path


## Link for the models :
[model link](https://drive.google.com/drive/folders/1QKUU0w7etRXGvFEKYSeKn5LTMVLIDlhm?usp=sharing)