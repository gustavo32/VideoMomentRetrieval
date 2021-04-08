from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.data import Dataset
from tensorflow.keras.layers import Layer, Bidirectional, Embedding, GRU
from tensorflow.keras import initializers

import pandas as pd
import numpy as np
from utils import utils


def get_pretrained_embeddings(embedding_dim, embedding_file):
    with open(embedding_file, 'rb') as f:
        data = [line.split(maxsplit=1) for line in f]
    pretrained_embeddings = pd.DataFrame.from_records(data, columns=['word', 'coefs'])
    pretrained_embeddings['word'] = pretrained_embeddings['word'].apply(lambda x: x.decode())
    pretrained_embeddings['coefs'] = pretrained_embeddings['coefs'].apply(lambda x: np.fromstring(x, "f", sep=" "))
    pretrained_embeddings = pretrained_embeddings.set_index('word', drop=True).iloc[:, 0]    
    return pretrained_embeddings


def get_embedding_matrix(descriptions, embedding_file, embedding_dim, n_tokens):    
    pretrained_embeddings = get_pretrained_embeddings(embedding_dim, embedding_file)
    
    text_ds = Dataset.from_tensor_slices(descriptions).batch(16)
    
    vectorizer = TextVectorization(max_tokens=n_tokens, output_sequence_length=25)
    vectorizer.adapt(text_ds)
    voc = vectorizer.get_vocabulary()
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    hits = 0
    misses = 0

    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = pretrained_embeddings[word]
            hits += 1
        except KeyError:
            misses += 1
            pass
        
    return vectorizer, embedding_matrix


class SentenceLayer(Layer):
    def __init__(self, embedding_matrix, configs=None):
        super(SentenceLayer, self).__init__()
        configs = utils.load_configs_if_none(configs)
        self.embedding_1 = Embedding(
            len(embedding_matrix),
            configs.sentence.embeddings_dim,
            input_length=25,
            embeddings_initializer=initializers.Constant(embedding_matrix),
            trainable=False
        )
        
        self.bigru_1 = Bidirectional(GRU(configs.n_features // 2, return_sequences=True))
        
    def call(self, inputs):
        x = self.embedding_1(inputs)
        return self.bigru_1(x)