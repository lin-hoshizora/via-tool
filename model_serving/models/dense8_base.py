import numpy as np
import cv2
from .utils.general import softmax, greedy_decode
from .utils.image import resize_h

class Dense8Base:
  """
  DenseNet8 Inference Class
  """
  def __init__(self, logger, height, **kwargs):
    self.logger = logger
    self.input_len = -1
    self.ratio = None
    self.height = height
    self.stride = self.height // 4

  def preprocess(self, img, nchw=True):
    self.ratio = self.height / img.shape[0]
    img_pad, img_resize = resize_h(img, h=self.input_shape[2], w=self.input_shape[3], logger=self.logger)
    if nchw:
      img_pad = img_pad.transpose(2, 0, 1)
    img_pad = img_pad[np.newaxis, ...].astype(np.float32)
    self.input_len = max(img_resize.shape[1] // self.stride, self.input_len)
    return img_pad

  def parse_result(self, probs, num_only):
    probs = probs.reshape(probs.shape[1], probs.shape[-1])
    if len(probs.shape) == 4:
      probs = probs.transpose(1, 0)
    if num_only:
      probs_num = np.zeros_like(probs)
      num_indices = [1,6,17,31,34,42,46,49,50,39, probs.shape[-1]-1]
      probs_num[:, num_indices] = probs[:, num_indices]
      probs = probs_num
    codes, probs, positions = greedy_decode(probs, self.input_len)
    positions = (positions * self.stride + self.stride // 2) / self.ratio
    return codes, probs, positions

  def infer_sync(self, img, num_only=False):
    raise NotImplementedError
