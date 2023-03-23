import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import os
def load_embedding_file(filepath):
    word2idx = {}
    embeddings = []
    with open(filepath, 'r') as f:
        line = f.readline()
        vocab_size, embedding_size = map(int, line.strip().split())
        print('Loading embeddings : {}'.format(filepath))
        for i in tqdm(range(vocab_size)):
            line = f.readline()
            line = line.strip().split()
            word = line[0]
            embedding = np.array(line[1:], dtype=np.float32)
            word2idx[word] = len(word2idx)
            embeddings.append(embedding)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx,idx2word, np.array(embeddings)


def tsne_plot(results,save_path):
        words = []
        embeds = []

        for res in results:
            embeds.append(res[1])
            words.append(res[0])
        
        tsne_model = TSNE(init='pca')
        res_embeds = tsne_model.fit_transform(embeds)

        x_axis_val = []
        y_axis_val = []
        for val in res_embeds:
            x_axis_val.append(val[0])
            y_axis_val.append(val[1]) 
        plt.figure(figsize=(10, 10)) 
        for i in range(len(x_axis_val)):
            plt.scatter(x_axis_val[i],y_axis_val[i])
            plt.annotate(words[i],
                        xy=(x_axis_val[i],y_axis_val[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')            
        plt.savefig(save_path)
        plt.show()

def plot_top10_words(embedding_file_path,word, save_path):
    word = word.lower()
    save_path = os.path.join(save_path, word+'.png')
    vocab,ind2vocab,embeddings=load_embedding_file(embedding_file_path)
    if word not in vocab:
        print('Word not in vocabulary')
        exit(1)
    word_index = vocab[word]
    word_embed = embeddings[word_index]
    res = {}
    print('Calculating cosine similarity :')
    for i in tqdm(range(len(embeddings))):
        if i!=word_index:
            res[i] = [1 - cosine(embeddings[i], word_embed), embeddings[i]]
    results = []
    for t in sorted(res.items(), key=lambda item: item[1][0], reverse=True)[0:10]:
        results.append([ind2vocab[t[0]], t[1][1]])
    print('Word:-', word)
    print('Words:-', end='\t')
    for res in results:
        print(res[0], end=', ')
    print()
    tsne_plot(results,save_path)

