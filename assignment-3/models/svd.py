from scipy.sparse import csr_matrix
import numpy as np
from sklearn.decomposition import TruncatedSVD

class SVD_W2V:
    def __init__(self, vocab, window,embedding_size):
        self.dim = len(vocab)
        self.vocab = vocab
        self.window_size = window
        self.embedding_size = embedding_size
        self.cooccurrence_matrix = np.zeros((self.dim, self.dim))

    def train(self,data):
        self.__build_cooccurrence_matrix(data)
        svd = TruncatedSVD(n_components=self.embedding_size, random_state=42)
        self.embeddings = svd.fit_transform(self.cooccurrence_matrix)



    def __build_cooccurrence_matrix(self, data):
        self.cooccurrence_matrix = np.zeros((self.dim, self.dim))
        for tokens in data:
            for pos,token in enumerate(tokens):
                if token not in self.vocab:
                    continue
                start = max(0, pos - self.window_size)
                end = min(len(tokens), pos + self.window_size)
                for context_pos in range(start, end):
                    if context_pos != pos:
                        context_token = tokens[context_pos]
                        if context_token in self.vocab:
                            self.cooccurrence_matrix[self.vocab[token], self.vocab[context_token]] += 1
                        else:
                            self.cooccurrence_matrix[self.vocab[token], self.vocab["<unk>"]] += 1
        self.cooccurrence_matrix = csr_matrix(self.cooccurrence_matrix)
    
    def save_embeddings(self, path):
        with open(path, 'w') as f:
            f.write('{} {}\n'.format(self.dim, self.embedding_size))
            for word, i in self.vocab.items():
                e = ' '.join(map(str, self.embeddings[i]))
                f.write('{} {}\n'.format(word, e))