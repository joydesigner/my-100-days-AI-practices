# Week 5 Challenge - Word2Vec KMeans Clustering

This project performs clustering on text data using pre-trained Word2Vec embeddings and the KMeans algorithm. The code processes a set of sentences, converts them into vectors, and groups them into clusters for further analysis.

## Features

- **Text Vectorization**: Uses a pre-trained Word2Vec model to convert sentences into vectors by averaging word embeddings.
- **KMeans Clustering**: Groups sentences into clusters using the KMeans algorithm.
- **Cluster Analysis**: Computes and displays the average distance within clusters and filters clusters based on distance thresholds.
- **Sentence Output**: Outputs representative sentences from each retained cluster.

## Requirements

- Python 3.x
- `jieba`: For Chinese text tokenization.
- `numpy`: For vector operations.
- `gensim`: For loading and working with Word2Vec models.
- `scikit-learn`: For KMeans clustering and distance calculations.

Install dependencies:
```bash
pip install jieba numpy gensim scikit-learn