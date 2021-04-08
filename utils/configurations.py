import os
import pickle
import shutil


class VideoConfigs():
    n_splits = 6
    n_extracted_features = 400
    max_frames = 900
    size = (224, 224)
    features_folder = 'visual_features_extracted/'
    files_pattern = 'datasets/DiDeMo/{}.mp4'


class SentenceConfigs():
    embeddings_folder = 'utils/vocabulary/'
    embeddings_dim = 50
    embedding_file = embeddings_folder + f'glove.6B.{embeddings_dim}d.txt'
    n_tokens = 20000
    
    def __setattr__(self, name, value):
        if name == 'embeddings_dim':
            self.__dict__['embedding_file'] = self.embeddings_folder + f'glove.6B.{value}d.txt'
        elif name == 'embeddings_folder':
            self.__dict__['embedding_file'] = name + f'glove.6B.{self.embeddings_dim}d.txt'
            
        self.__dict__[name] = value

        
class DatasetConfigs():
    video = VideoConfigs()
    sentence = SentenceConfigs()
#     audio = AudioConfigs()
    fps = 30
    clip_length = 150
    train_info_path = 'data/train_data.json'
    valid_info_path = 'data/val_data.json'
    test_info_path = 'data/test_data.json'
    batch_size = 16
    include_audio_features = False
    n_features = 128
    
    def __enter__(self):
        if not os.path.isdir('temp/'):
            os.makedirs('temp/', exist_ok=True)
        with open('temp/configs.pkl', 'wb') as f:
            pickle.dump(self, f)
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree('temp/')
    
    def describe(self, __obj=None):
        if __obj is None:
            __obj = self
            
        for attr in dir(__obj):
            if attr[:2] != '__' or attr[-2:] != '__':
                value = getattr(__obj, attr)
                if hasattr(value, '__call__'):
                    continue
                
                if hasattr(value, '__dict__'):
                    print(attr)
                    self.describe(value)
                    continue
                
                if __obj != self:
                    print('  --- ', end='')
                print(attr, '=', getattr(__obj, attr))