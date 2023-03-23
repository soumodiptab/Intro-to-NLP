#!/bin/bash

cbow_file="embeddings/cbow_neg/embeddings_300.txt"
svd_file="embeddings/svd/svd_embeddings_300.txt"
words="titanic captain ship crew boat ocean sea"
echo "Plotting CBOW embeddings"
for word in $words
do
    python3 main.py plot $cbow_file $word cbow_neg
done
echo "Plotting SVD embeddings"
for word in $words
do
    python3 main.py plot $svd_file $word svd
done