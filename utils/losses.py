import numpy as np
import tensorflow as tf


def get_ranking_loss(scores, margin, top_k=8):
    scores_positives = tf.linalg.diag_part(scores)

    shape_negative = list(scores.shape)
    bool_matrix = tf.ones(shape_negative)
    bool_matrix = tf.linalg.set_diag(bool_matrix, tf.zeros(shape_negative[0])) == 1
    shape_negative[-1] = shape_negative[-1] - 1

    scores_negatives = tf.reshape(scores[bool_matrix], shape_negative)
    top_k_scores_negatives = tf.sort(scores_negatives, axis=1, direction="DESCENDING")[:, 0]

    loss = margin - scores_positives + top_k_scores_negatives
    loss = tf.where(loss < 0.0, 0.0, loss)

    return tf.reduce_sum(loss)


def old_margin_based_ranking_loss(scores_videos, scores_sentences, margin=1, top_k=8):
    video_loss = get_ranking_loss(scores_videos, margin, top_k)
    sentence_loss = get_ranking_loss(scores_sentences, margin, top_k)
    
    return video_loss + sentence_loss


def margin_based_ranking_loss(positive, negative, margin=1, top_k=8):
    return tf.reduce_max([16 - tf.reduce_sum(positive) + tf.reduce_sum(negative), 0])


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
