from utils import configurations, utils
from moviepy.editor import VideoFileClip
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers import Layer, Bidirectional, GRU
import numpy as np
import os


def get_frames_properly_formatted(path, size, max_frames):
    clip = VideoFileClip(path)
    resized_clip = clip.resize(size)
    transf_frames = np.asarray(list(resized_clip.iter_frames())) / 255.0
    transf_frames = transf_frames[:max_frames]
    
    remaining_frames = max_frames - len(transf_frames)
    if remaining_frames > 0:
        zeros = np.zeros_like(transf_frames)
        transf_frames = np.append(transf_frames, zeros[:remaining_frames], axis=0)
    
    return transf_frames


def extract_visual_features(frames, model, n_splits):
    video_features = []
    for n_array in np.split(frames, n_splits):
        video_features.append(model(tf.convert_to_tensor(n_array[None], dtype=float))['default'][0].numpy())

    return np.asarray(video_features)


def _generator_preprocess_video_dataset(videos_path, features_folder, size, n_splits, max_frames):
    i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']
    
    for path in videos_path:
        features_path = (features_folder+os.path.split(path)[-1]).decode()
        path = path.decode()
        if not os.path.isfile(features_path):
            frames = get_frames_properly_formatted(path, size, max_frames)
            features = extract_visual_features(frames, i3d, n_splits)
            np.save(features_path, features)
        else:
            features = np.load(features_path)
        
        yield np.asarray(features).astype('float32')
        

def preprocess_video_dataset(info, configs=None):
    configs = utils.load_configs_if_none(configs)
    return tf.data.Dataset.from_generator(
        _generator_preprocess_video_dataset,
        output_types=tf.dtypes.float32,
        output_shapes=(configs.video.n_splits, configs.video.n_extracted_features),
        args = (
            info['video'].values.tolist(),
            configs.video.features_folder,
            configs.video.size,
            configs.video.n_splits,
            configs.video.max_frames
        )
    )


class VideoLayer(Layer):
    def __init__(self, configs=None):
        super(VideoLayer, self).__init__()
        configs = utils.load_configs_if_none(configs)
        self.bigru_1 = Bidirectional(GRU(configs.n_features // 2, return_sequences=True))
        
    def call(self, inputs):
        return self.bigru_1(inputs)