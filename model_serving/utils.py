import logging
import sys
from datetime import datetime
import yaml
import numpy as np


def get_logger(config):
  formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
  logger = logging.getLogger()
  logger.handlers = []
  logger.setLevel(getattr(logging, config.get('level', 'DEBUG')))

  # stream
  stream_handler = logging.StreamHandler(stream=sys.stdout)
  stream_handler.setFormatter(formatter)
  logger.addHandler(stream_handler)

  # file
  log_path = config.get('path', None)
  if log_path is not None:
    log_folder = Path(__file__).parent.parent / log_path
    if not log_folder.exists():
      log_folder.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(f"{log_folder}/{get_timestamp()}_scanner.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
  return logger

def decode_img(pb):
  img = np.frombuffer(pb.data, np.uint8)
  img = img.reshape(pb.h, pb.w, pb.c)
  return img

def get_timestamp():
  timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
  return timestamp

def get_dense_key(img):
  w_resized = 64 / img.shape[0] * img.shape[1]
  if w_resized > 1024:
    return 1408
  if w_resized > 192:
    return 1024
  return 192

def get_layout(img):
  layout = 'portrait' if img.shape[0] > img.shape[1] else 'landscape'
  return layout

def pad1d(x, length, constant):
  padded = np.pad(x, ((0, length - x.shape[0])), constant_values=constant)
  return padded

def load_conf(path):
  with open(path, encoding='utf-8') as f:
    res = yaml.safe_load(f)
  return res

try:
  import pyarmnn as nn
  class ArmNNRuntime:
    def __init__(self):
      options = ann.CreationOptions()
      self.runtime = ann.IRuntime(options)
      self.infer_idx = 0

    def get_infer_idx(self):
      prev = self.infer_idx
      self.infer_idx += 1
      return prev
except ModuleNotFoundError:
  print('PyArmNN is not installed')
