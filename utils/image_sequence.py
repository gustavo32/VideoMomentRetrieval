from utils import configurations, utils
from moviepy.editor import VideoFileClip, ColorClip, concatenate_videoclips
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers import Layer, Bidirectional, GRU, Dense
import numpy as np
import os
from tqdm import tqdm


def get_frames_properly_formatted(path, size, max_frames, fps):
    clip = VideoFileClip(path, target_resolution=size, verbose=False).set_fps(fps)
    duration = max_frames - clip.duration * clip.fps
    if duration <= 0:
        clip = clip.set_duration(max_frames//fps)
    else:
        blank_video = ColorClip(size, (0, 0, 0), duration=duration/fps)
        clip = concatenate_videoclips([clip, blank_video])
    video = np.asarray(list(clip.iter_frames())) / 255.0
    return video[:max_frames]


def extract_visual_features_from_sample(frames, model, n_splits):
    video_features = []
    for n_array in np.split(frames, n_splits):
        video_features.append(model(tf.convert_to_tensor(
            n_array[None], dtype=float))['default'][0].numpy())

    return np.asarray(video_features)


def extract_visual_features(info, configs): 
    i3d = hub.load(
            "https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

    for row in tqdm(info.itertuples()):
        try:
            if not os.path.isfile(row.features_path):
                    frames = get_frames_properly_formatted(row.video, configs.video.size, configs.video.max_frames, configs.fps)
                    features = extract_visual_features_from_sample(frames, i3d, configs.video.n_splits)
                    np.save(row.features_path, features)
        except OSError:
            with open("videos_error.txt", "a+") as f:
                f.write(row.video+"\n")


class VideoLayer(Layer):
    def __init__(self, configs=None):
        super(VideoLayer, self).__init__()
        configs = utils.load_configs_if_none(configs)
        self.dense_1 = Dense(configs.n_features)
#         self.bigru_1 = Bidirectional(
#             GRU(configs.n_features // 2, return_sequences=True))

    def call(self, inputs):
        return self.dense_1(inputs)
