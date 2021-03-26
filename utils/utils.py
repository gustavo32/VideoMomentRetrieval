import base64
import json
import mimetypes
import os
import pathlib
import pickle
import textwrap

import imageio
import IPython.display
import numpy as np
import pandas as pd


def embed_data(mime: str, data: bytes) -> IPython.display.HTML:
    """Embeds data as an html tag with a data-url."""
    b64 = base64.b64encode(data).decode()
    if mime.startswith('image'):
        tag = f'<img src="data:{mime};base64,{b64}"/>'
    elif mime.startswith('video'):
        tag = textwrap.dedent(f"""
            <video width="640" height="480" controls>
              <source src="data:{mime};base64,{b64}" type="video/mp4">
              Your browser does not support the video tag.
            </video>
            """)
    else:
        raise ValueError('Images and Video only.')
    return IPython.display.HTML(tag)


def embed_file(path: os.PathLike) -> IPython.display.HTML:
    """Embeds a file in the notebook as an html tag with a data-url."""
    path = pathlib.Path(path)
    mime, unused_encoding = mimetypes.guess_type(str(path))
    data = path.read_bytes()

    return embed_data(mime, data)


def to_gif(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=30)
    return embed.embed_file('./animation.gif')


def load_configs_if_none(configs):
    if configs is None:
        with open('temp/configs.pkl', 'rb') as f:
            configs = pickle.load(f)
    return configs


def most_agreed_labels(labels):
    str_labels = [str(start) + ':' + str(end) for start, end in labels]
    uniques, idx, counts = np.unique(
        str_labels, return_counts=True, return_index=True)
    idx_max = np.argmax(counts)
    agreed_labels = labels[idx[idx_max]]
    return pd.Series(agreed_labels)