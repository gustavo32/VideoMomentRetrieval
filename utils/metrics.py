import numpy as np


def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window


def each_pred_moving_average(scores, period):
    length = len(scores)
    repeated_scores = np.tile(scores, 2)[length-period+1:]
    ma = moving_average(repeated_scores, period)
    argmax_ma = np.argmax(ma)
    individual_proposals = ma > scores.mean()
    breaks_on_false = individual_proposals & np.roll(individual_proposals, 1)
    indices = np.arange(len(breaks_on_false))
    positions_to_break = np.where(breaks_on_false == False, indices, -1)[1:]
    positions_to_break = positions_to_break[positions_to_break != -1]

    for segment in np.split(indices, positions_to_break):
        if argmax_ma in segment:
            break

    return segment[0], segment[-1]


def prediction_by_moving_average(scores_videos, period=3):
    return np.apply_along_axis(each_pred_moving_average, 1, scores_videos, period=period)


def intersection_over_union(labels):
    start_true, end_true, start_pred, end_pred = labels
    y_true = np.arange(start_true, end_true+1)
    y_pred = np.arange(start_pred, end_pred+1)
    intersection = np.intersect1d(y_true, y_pred)
    union = np.union1d(y_true, y_pred)

    return len(intersection)/len(union)


def mIOU(scores_videos, y_true):
    y_pred = prediction_by_moving_average(scores_videos)
    scores = np.apply_along_axis(
        intersection_over_union, 1, np.concatenate([y_true, y_pred], axis=-1))
    return scores.mean()
