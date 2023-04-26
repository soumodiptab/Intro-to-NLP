import torch
import numpy as np
from tqdm import tqdm
def load_embeddings(vocab,embeddings_file,dimension):
    # load only the embeddings that are in the vocab
    embeddings = np.zeros((len(vocab), dimension))
    with open(embeddings_file, 'r') as f:
        for line in tqdm(f):
            line = line.split()
            word = line[0]
            if word in vocab:
                embeddings[vocab[word]] = np.array(line[1:], dtype=np.float32)
    return torch.Tensor(embeddings)