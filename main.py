from utils import configurations, base, image_sequence, sentence
import tensorflow as tf

configs = configurations.DatasetConfigs()
configs.n_features = 64
configs.describe()

with configs:
    train_ds, valid_ds, test_ds, embedding_matrix = base.preprocess_datasets()

    video_layer = image_sequence.VideoLayer()
    sentence_layer = sentence.SentenceLayer(embedding_matrix)
    #     audio_layer = get_audio_layer()

    moment = base.MomentVideo(video_layer, sentence_layer, configs.batch_size)

tf.config.set_soft_device_placement(True)
moment.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))
with tf.device("GPU:0"):
    moment.fit(train_ds.shuffle(100).cache(), epochs=50, validation_data=valid_ds)

moment.evaluate(test_ds)