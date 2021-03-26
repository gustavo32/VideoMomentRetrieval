from utils import utils, image_sequence, sentence
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import convert_to_tensor, split, expand_dims
from utils.losses import margin_based_ranking_loss
from utils.metrics import mIOU, prediction_by_moving_average
from time import time

def preprocess_dataset(configs=None):
    configs = utils.load_configs_if_none(configs)
    
    with open(configs.info_path, 'r') as f:
        info = pd.Series(json.load(f)).apply(pd.Series)
    
    info['video'] = info['video'].apply(lambda x: configs.video.files_pattern.format(x))
    
    labels = info['times'].apply(utils.most_agreed_labels)
    
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    
    videos_dataset = image_sequence.preprocess_video_dataset(info, configs)
    sentences_dataset, embedding_matrix = sentence.preprocess_sentence_dataset(info, configs)
    
    if configs.include_audio_features:
        audios_dataset = preprocess_audio_dataset()
        zip_datasets_list = [videos_dataset, sentences_dataset, audio_dataset, labels_dataset]
    else:
        zip_datasets_list = [videos_dataset, sentences_dataset, labels_dataset]
        
    dataset = tf.data.Dataset.zip(tuple(zip_datasets_list))\
                .batch(configs.batch_size, drop_remainder=True)
    
    return dataset, embedding_matrix


class MomentVideo(Model):
    def __init__(self, video_layer, sentence_layer, batch_size=8):
        super(MomentVideo, self).__init__()
        self.video_1 = video_layer
        self.sentence_1 = sentence_layer
        self.batch_size = batch_size
    
    
    def cosine_similarity(self, tensor1, tensor2):
        num = tf.linalg.matmul(expand_dims(tensor1, 1), expand_dims(tensor2, 1), transpose_a=True)
        den = tf.norm(tensor1)*tf.norm(tensor2)
        return num/(den+1e-15)
    
    
    def matrix_cosine_similarity(self, tensor1, tensor2):
        matrix = []
        for i in range(tensor1.shape[0]):
            row = []
            for j in range(tensor2.shape[0]):
                row.append(self.cosine_similarity(tensor1[i, :], tensor2[j, :]))
            matrix.append(row)
        return tf.stack(matrix)[:,:,0,0]
    
    
    def similarity_between_repr_and_attend(self, tensor1, tensor2):
        scores = []
        for k in range(tensor1.shape[0]):
            scores.append(self.cosine_similarity(tensor1[k, :], tensor2[k, :]))
        return tf.stack(scores)[:, 0, 0]
    
    
    def get_scores(self, video_repr_tensor, sentence_repr_tensor):
        coattention_matrix = self.matrix_cosine_similarity(video_repr_tensor, sentence_repr_tensor)
        
        normalized_sentence = tf.nn.softmax(coattention_matrix, axis=1)
        normalized_video = tf.nn.softmax(coattention_matrix, axis=0)
            
        matrix = []
        for i in range(video_repr_tensor.shape[0]):
            row_sum = np.zeros((128))
            for j in range(sentence_repr_tensor.shape[0]):
                row_sum += normalized_sentence[i][j] * sentence_repr_tensor[j, :]
            matrix.append(row_sum)
            
        sentence_attention = tf.stack(matrix)
    
        matrix = []
        for j in range(sentence_repr_tensor.shape[0]):
            row_sum = np.zeros((128))
            for i in range(video_repr_tensor.shape[0]):
                row_sum += normalized_video[i][j] * video_repr_tensor[i, :]
            matrix.append(row_sum)
            
        video_attention = tf.stack(matrix)
        
        scores_video = self.similarity_between_repr_and_attend(video_repr_tensor, sentence_attention)
        scores_sentence = self.similarity_between_repr_and_attend(sentence_repr_tensor, video_attention)
        
        return scores_video, scores_sentence
    
    
    def call(self, data, training=False):

        videos, sentences = data
        
        start_time = time()
        
        videos_repr = self.video_1(videos)
        sentences_repr = self.sentence_1(sentences)
# #         print(videos_repr[(videos_repr == -np.inf) | (videos_repr == np.inf) | (pd.isna(videos_repr))])
        print("Time to get the representations:", time()-start_time)
    
        if training == False:
            scores_videos = []
            scores_sentences = []
            for i in range(self.batch_size):
                scores_video, scores_sentence = self.get_scores(videos_repr[i], sentences_repr[i])
                scores_videos.append(scores_video)
                scores_sentences.append(scores_sentence)
            
            scores_videos = np.vstack(scores_videos)
            scores_sentences = np.vstack(scores_sentences)
            
            return scores_videos, scores_sentences
        
        else:
            sum_scores_videos = []
            sum_scores_sentences = []
            for i in range(self.batch_size):
                _sum_scores_videos = []
                _sum_scores_sentences = []
                for j in range(self.batch_size):
                    start_time = time()
                    scores_video, scores_sentence = self.get_scores(videos_repr[i], sentences_repr[j])
                    print("time to compute the score between video and sentence:", time()-start_time)
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
            print(videos.shape)
            with tf.GradientTape() as tape:
                print("before_training")
                scores_videos, scores_sentences = self((videos, sentences), training=True)  # Forward pass
                print("after_training")
                # Compute the loss margin-based ranking loss
                loss = margin_based_ranking_loss(scores_videos, scores_sentences, margin=1, top_k=2)
                print("loss", loss)
                
            tf.print(loss)
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#             # Compute our own metrics
#             loss_tracker.update_state(loss)
#             mae_metric.update_state(y, y_pred)
            print("loss", loss)
            return {"loss": loss}

    def test_step(self, data):
        print('test')
        (videos, sentences), y_true = data
        scores_videos, scores_sentences = self((videos, sentences), training=False)
        miou_score = tf.numpy_function(mIOU, [scores_videos, y_true], tf.float32)
        return {'mIOU': miou_score}