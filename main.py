from utils import configurations, base, image_sequence, sentence
import tensorflow as tf

if __name__ == "__main__":

    configs = configurations.DatasetConfigs()
    configs.n_features = 128
    configs.describe()

    with configs:
        train_ds, valid_ds, test_ds, embedding_matrix = base.preprocess_datasets()

        video_layer = image_sequence.VideoLayer()
        sentence_layer = sentence.SentenceLayer(embedding_matrix)
        #     audio_layer = get_audio_layer()

        moment = base.MomentVideo(video_layer, sentence_layer, configs.batch_size)

    tf.config.set_soft_device_placement(True)
    moment.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))
    moment.evaluate(test_ds)
    with tf.device("GPU:0"):
        moment.fit(train_ds.shuffle(100).cache(), epochs=3, validation_data=valid_ds)

    print(moment.evaluate(test_ds))
