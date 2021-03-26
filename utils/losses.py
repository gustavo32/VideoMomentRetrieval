import numpy as np
import tensorflow as tf


def get_ranking_loss(scores, margin, top_k=2):
    shape_negative = list(scores.shape)
    shape_negative[-1] = shape_negative[-1] - 1
    
    scores_true = tf.linalg.diag_part(scores)
    scores_negatives = tf.sort(tf.reshape(scores[scores != tf.transpose(scores)], shape_negative), axis=1)[:, ::-1][:, :top_k]
               
    loss = margin - tf.tile(tf.reshape(scores_true, (-1, 1)), [1, scores_negatives.shape[1]]) + scores_negatives
    loss = tf.where(loss < 0.0, 0.0, loss)
    loss = tf.reduce_sum(loss, axis=1)

    return loss


def margin_based_ranking_loss(scores_videos, scores_sentences, margin=1, top_k=2):
    
    video_loss = get_ranking_loss(scores_videos, margin, top_k)
    sentence_loss = get_ranking_loss(scores_sentences, margin, top_k)
    
    return video_loss + sentence_loss


def numpy_margin_based_ranking_loss(scores_video, scores_sentence, margin=1, top_k=2):
    
    shape_negative = list(scores_video.shape)
    shape_negative[-1] = shape_negative[-1] - 1

    score_true_videos = scores_video.diagonal()
    scores_negatives_videos = np.sort(scores_video[scores_video != scores_video.T].reshape(shape_negative), axis=1)[:, ::-1][:, :top_k]
    
    video_loss = 1 - np.tile(score_true_videos.reshape(-1, 1), scores_negatives_videos.shape[1]) + scores_negatives_videos
    video_loss = np.where(video_loss < 0, 0, video_loss)
    video_loss = np.sum(video_loss, axis=1)

    shape_negative = list(scores_sentence.shape)
    shape_negative[-1] = shape_negative[-1] - 1

    score_true_sentences = scores_sentence.diagonal()
    scores_negatives_sentences = np.sort(scores_sentence[scores_sentence != scores_sentence.T].reshape(shape_negative), axis=1)[:, ::-1][:, :top_k]

    sentence_loss = 1 - np.tile(score_true_sentences.reshape(-1, 1), scores_negatives_sentences.shape[1]) + scores_negatives_sentences
    sentence_loss = np.where(sentence_loss < 0, 0, sentence_loss)
    sentence_loss = np.sum(sentence_loss, axis=1)

    return video_loss + sentence_loss
