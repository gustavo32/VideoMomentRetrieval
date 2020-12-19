import cv2
import numpy as np
import tensorflow as tf

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]


def load_transform_video(path, max_frames=900, min_frames=900, resize=(112, 112)):
    cap = cv2.VideoCapture(path)
    frames = np.empty(resize+tuple([3]))[None]
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]][None]
            frames = np.vstack([frames, frame])

            if len(frames) == max_frames:
                break
            
        while len(frames) < min_frames or (len(frames) > min_frames and len(frames) < max_frames):
            frames = np.vstack([frames, np.zeros(resize + tuple([3]))[None]])
        
    finally:
        cap.release()
        
    return (frames / 255.0).astype(np.float32)

def save_video(series, fps=30, size=(112,112)):
    out = cv2.VideoWriter(series['path_out'], cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(series['frames'])):
        # writing to a image array
        out.write(series['frames'][i])
    out.release()

# def mapped_load_transform_video(paths, max_frames, min_frames, num_parallel_calls):
#     return paths.map(lambda path: tf.numpy_function(load_transform_video, [path], [tf.float32]), num_parallel_calls=num_parallel_calls)