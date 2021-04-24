from time import time
import pandas as pd
import numpy as np
import json
import os
import h5py
from itertools import product

from tensorflow import convert_to_tensor, split, expand_dims
from tensorflow.keras import Model
import tensorflow as tf

from utils.metrics import mIOU, prediction_by_moving_average
from utils.losses import margin_based_ranking_loss
from utils import utils, image_sequence, sentence


def generator_files(paths):
    for path in paths:
        features = None
        with h5py.File(path, "r") as hf:
            features = hf["features"]
            remaining_frames = 150 - features.shape[0]
            pad_features = tf.concat([hf["features"], tf.zeros([remaining_frames, features.shape[1]])], axis=0)
        yield pad_features
            

def preprocess_dataset(dataset_path, configs=None):
    configs = utils.load_configs_if_none(configs)

    with open(dataset_path, 'r') as f:
        info = pd.Series(json.load(f)).apply(pd.Series)

    vectorizer, embedding_matrix = sentence.get_embedding_matrix(
        info['description'],
        configs.sentence.embedding_file,
        configs.sentence.embeddings_dim,
        configs.sentence.n_tokens
    )

    info['features_path'] = info['video'].apply(lambda x: configs.video.features_folder + "fc7_subsample5_fps25_" + x + ".h5")

    labels_dataset = tf.data.Dataset.from_tensor_slices(info['times'].apply(utils.most_agreed_labels).values)
    videos_dataset = tf.data.Dataset.from_generator(generator_files, args=[info["features_path"].values.tolist()],
                                   output_types=tf.float32, output_shapes = (150, 4096))
    descriptions = vectorizer(info['description'])
    lengths_dataset = tf.data.Dataset.from_tensor_slices(tf.math.count_nonzero(descriptions, axis=1))
    sentences_dataset = tf.data.Dataset.from_tensor_slices(descriptions)

    if configs.include_audio_features:
        audios_dataset = preprocess_audio_dataset()
        zip_datasets_list = (videos_dataset, sentences_dataset, lengths_dataset, audio_dataset, labels_dataset)
    else:
        zip_datasets_list = (videos_dataset, sentences_dataset, lengths_dataset, labels_dataset)

    dataset = tf.data.Dataset.zip(zip_datasets_list).batch(configs.batch_size, drop_remainder=True)
    return dataset, embedding_matrix


def preprocess_datasets(configs=None):
    configs = utils.load_configs_if_none(configs)
    
    train_ds, embedding_matrix = preprocess_dataset(configs.train_info_path, configs)
    valid_ds, _ = preprocess_dataset(configs.valid_info_path, configs)
    test_ds, _ = preprocess_dataset(configs.test_info_path, configs)
    
    return train_ds, valid_ds, test_ds, embedding_matrix
    

class MomentVideo(Model):
    def __init__(self, video_layer, sentence_layer, proposals, batch_size=8):
        super(MomentVideo, self).__init__()
        self.video_1 = video_layer
        self.sentence_1 = sentence_layer
        self.batch_size = batch_size
        self.attn_layer = tf.keras.layers.Attention()
        self.proposals = proposals
    
    def cosine_similarity(self, tensor1, tensor2, axis=2):
        num = tf.reduce_sum(tensor1 * tensor2, axis=axis)
        den = tf.norm(tensor1, axis=axis) * tf.norm(tensor2, axis=axis)
        return (num/(den+1e-15))
    
    
    def attention(self, query, context, queryL, sourceL, smooth=1., axis=2):
        batch_size = query.shape[0]

        queryT = tf.transpose(query, [0, 2, 1])
        attn = tf.matmul(context, queryT)
        
        attn = tf.keras.layers.LeakyReLU(0.1)(attn)
        attn = tf.math.l2_normalize(attn, axis=axis)
        
        attn = tf.transpose(attn, [0, 2, 1])
        attn = tf.reshape(attn, (batch_size*queryL, sourceL))
        attn = tf.nn.softmax(attn*smooth)
        attn = tf.reshape(attn, (batch_size, queryL, sourceL))
        attnT = tf.transpose(attn, [0, 2, 1])
        contextT = tf.transpose(context, [0, 2, 1])
        weighted_context  = tf.matmul(contextT, attnT)
        weighted_context = tf.transpose(weighted_context, [0, 2, 1])

        return weighted_context
    
    
    @tf.function
    def call(self, data):
        videos, sentences, y_true = data

        videos_repr = self.video_1(videos)
        sentences_repr = self.sentence_1(sentences)

        batch_positive_score = []
        batch_negative_score = []
        for i in range(self.batch_size):
            positive_score = 0.0
            negative_scores = []
            for proposal in self.proposals:
                attn_video = self.attn_layer([sentences_repr[i][None], videos_repr[i, proposal[0]*25:(proposal[1]+1)*25]])
                attn_video = attn_video[0]
                score = self.cosine_similarity(attn_video, sentences_repr[i], axis=0)
                if tf.reduce_all(proposal == y_true[i]):
                    positive_score = score
                else:
                    negative_scores.append(score)

            batch_positive_score.append(positive_score)
            batch_negative_score.append(tf.reduce_max(negative_scores))

        return tf.stack(batch_positive_score), tf.stack(batch_negative_score)
    
    
    def train_step(self, data):
        videos, sentences, lengths, y_true = data
        with tf.GradientTape() as tape:
            positive, negative = self((videos, sentences, y_true))  # Forward pass
            # Compute the loss margin-based ranking loss
            loss = margin_based_ranking_loss(positive, negative, margin=0.7, top_k=8)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#             # Compute our own metrics
#             loss_tracker.update_state(loss)
#             mae_metric.update_state(y, y_pred)
        return {"loss": loss}

    
    @tf.function
    def test_step(self, data):
        videos, sentences, lengths, y_true = data
        positive, negative = self((videos, sentences, y_true), training=False)
        loss = margin_based_ranking_loss(positive, negative, margin=0.7, top_k=8)
        
        return {'loss': loss}