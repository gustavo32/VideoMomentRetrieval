from time import time
import pandas as pd
import numpy as np
import json
import os

from tensorflow import convert_to_tensor, split, expand_dims
from tensorflow.keras import Model
import tensorflow as tf

from utils.metrics import mIOU, prediction_by_moving_average
from utils.losses import margin_based_ranking_loss
from utils import utils, image_sequence, sentence


def preprocess_dataset(dataset_path, configs=None):
    configs = utils.load_configs_if_none(configs)
    
    with open(dataset_path, 'r') as f:
        info = pd.Series(json.load(f)).apply(pd.Series)

    info['video'] = info['video'].apply(lambda x: configs.video.files_pattern.format(x))
    info['features_path'] = info['video'].apply(lambda x: configs.video.features_folder+os.path.split(x)[-1] + '.npy')
    error_videos = info['video'].isin(pd.read_csv("videos_error.txt", header=None).iloc[:, 0])
    info = info[~error_videos.values]

    image_sequence.extract_visual_features(info.drop_duplicates(subset=['video']), configs)
    vectorizer, embedding_matrix = sentence.get_embedding_matrix(
        info['description'],
        configs.sentence.embedding_file,
        configs.sentence.embeddings_dim,
        configs.sentence.n_tokens
    )

    labels_dataset = info['times'].apply(utils.most_agreed_labels).values
    videos_dataset = np.stack(info['features_path'].apply(np.load).values)
    sentences_dataset = vectorizer(info['description']).numpy()
    
    if configs.include_audio_features:
        audios_dataset = preprocess_audio_dataset()
        zip_datasets_list = (videos_dataset, sentences_dataset, audio_dataset, labels_dataset)
    else:
        zip_datasets_list = (videos_dataset, sentences_dataset, labels_dataset)
        
    
    dataset = tf.data.Dataset.from_tensor_slices(zip_datasets_list)\
                .batch(configs.batch_size, drop_remainder=True)
    
    return dataset, embedding_matrix


def preprocess_datasets(configs=None):
    configs = utils.load_configs_if_none(configs)
    
    train_ds, embedding_matrix = preprocess_dataset(configs.train_info_path, configs)
    valid_ds, _ = preprocess_dataset(configs.valid_info_path, configs)
    test_ds, _ = preprocess_dataset(configs.test_info_path, configs)
    
    return train_ds, valid_ds, test_ds, embedding_matrix
    

class MomentVideo(Model):
    def __init__(self, video_layer, sentence_layer, batch_size=8):
        super(MomentVideo, self).__init__()
        self.video_1 = video_layer
        self.sentence_1 = sentence_layer
        self.batch_size = batch_size
    
    @tf.function
    def cosine_similarity(self, tensor1, tensor2):
        num = tf.reduce_sum(tensor1 * tensor2, axis=1)
        den = tf.norm(tensor1, axis=1) * tf.norm(tensor2, axis=1)
        return (num/(den+1e-15))
    
    @tf.function
    def matrix_cosine_similarity(self, tensor1, tensor2):
        repeated_tensor1 = tf.repeat(tensor1, repeats=tensor2.shape[0], axis=0)
        tiled_tensor2 = tf.tile(tensor2, tf.constant([tensor1.shape[0], 1], dtype=tf.int32))
        similarity = self.cosine_similarity(repeated_tensor1, tiled_tensor2)
        return tf.reshape(similarity, [tensor1.shape[0], tensor2.shape[0]])
    
    @tf.function
    def get_scores(self, video_repr_tensor, sentence_repr_tensor):
        coattention_matrix = self.matrix_cosine_similarity(video_repr_tensor, sentence_repr_tensor)
        
        normalized_sentence = tf.nn.softmax(coattention_matrix, axis=1)
        normalized_video = tf.nn.softmax(coattention_matrix, axis=0)
            
        sentence_attention = tf.linalg.matmul(normalized_sentence, sentence_repr_tensor)
        video_attention = tf.linalg.matmul(normalized_video, video_repr_tensor, transpose_a=True)
        
        scores_video = self.cosine_similarity(video_repr_tensor, sentence_attention)
        scores_sentence = self.cosine_similarity(sentence_repr_tensor, video_attention)
        
        return scores_video, scores_sentence
    
    
    def call(self, data, training=False):

        videos, sentences = data
        
        videos_repr = self.video_1(videos)
        sentences_repr = self.sentence_1(sentences)
    
        if training == False:
            scores_videos = []
            scores_sentences = []
            for i in range(self.batch_size):
                scores_video, scores_sentence = self.get_scores(videos_repr[i], sentences_repr[i])
                scores_videos.append(scores_video)
                scores_sentences.append(scores_sentence)
            
            scores_videos = tf.stack(scores_videos)
            scores_sentences = tf.stack(scores_sentences)
            
            return scores_videos, scores_sentences
        
        else:
            sum_scores_videos = []
            sum_scores_sentences = []
            for i in range(self.batch_size):
                _sum_scores_videos = []
                _sum_scores_sentences = []
                for j in range(self.batch_size):
                    scores_video, scores_sentence = self.get_scores(videos_repr[i], sentences_repr[j])
                    _sum_scores_videos.append(tf.reduce_sum(scores_video)/len(scores_video))
                    _sum_scores_sentences.append(tf.reduce_sum(scores_sentence)/len(scores_sentence))
                
                sum_scores_videos.append(_sum_scores_videos)
                sum_scores_sentences.append(_sum_scores_sentences)
            
            sum_scores_videos = tf.stack(sum_scores_videos)
            sum_scores_sentences = tf.transpose(tf.stack(sum_scores_sentences))
            
            return sum_scores_videos, sum_scores_sentences
    
    
    def train_step(self, data):
            videos, sentences, y_true = data
#             videos, sentences, y_true = [(v, s, l) for (v, s, l) in data.take(1)]
            with tf.GradientTape() as tape:
                scores_videos, scores_sentences = self((videos, sentences), training=True)  # Forward pass
                # Compute the loss margin-based ranking loss
                loss = margin_based_ranking_loss(scores_videos, scores_sentences, margin=1, top_k=8)
                
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#             # Compute our own metrics
#             loss_tracker.update_state(loss)
#             mae_metric.update_state(y, y_pred)
            return {"loss": loss}

    def test_step(self, data):
        videos, sentences, y_true = data
        scores_videos, scores_sentences = self((videos, sentences), training=False)
        miou_score = tf.numpy_function(mIOU, [scores_videos, y_true], tf.float64)
        
        scores_videos, scores_sentences = self((videos, sentences), training=True)
        loss = margin_based_ranking_loss(scores_videos, scores_sentences, margin=1, top_k=8)
        
        return {'mIOU': miou_score, 'loss': loss}