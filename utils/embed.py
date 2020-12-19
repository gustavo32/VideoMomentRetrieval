"""Simple functions for embedding data in a notebook file."""

import base64
import mimetypes
import os
import pathlib
import textwrap

import IPython.display


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