import logging
import numpy as np


def softmax(x: np.ndarray):
  """
  Apply softmax on each row
  :param x: 2D array
  :return: 2D array after softmax
  """
  assert len(x.shape) == 2, 'Softmax expects 2D arrays'
  exp_x = np.exp(x - x.max(axis=1, keepdims=True))
  softmax_x = exp_x / exp_x.sum(axis=1, keepdims=True)
  return softmax_x

def greedy_decode(x: np.ndarray, length: int):
  """
  CTC greedy decoder
  :param x: CTC encoded sequence, last label as void
  :param length: sequence length
  :return: decoded sequence and probability for each char
  """
  lb_void = x.shape[1] - 1
  encodes = x.argmax(axis=1)
  probs = [x[r][i] for r, i in enumerate(encodes)]
  decodes = []
  dec_prob = []
  positions = []
  prev = -1
  for i, code in enumerate(encodes[:length]):
    if code != lb_void:
      if prev == lb_void or code != prev:
        decodes.append(code)
        dec_prob.append(probs[i])
        positions.append(i)
      else:
        if probs[i] > dec_prob[-1]:
          dec_prob[-1] = probs[i]
    prev = code
  decodes = np.array(decodes)
  dec_prob = np.array(dec_prob)
  positions = np.array(positions)
  return decodes, dec_prob, positions

def group_lines(texts: list, iou_threshold: float = 0.4):
  grouped = []
  texts = sorted(texts, key=lambda x: (x[-1][1] + x[-1][3]) / 2)
  current_line = []
  for text in texts:
    if not current_line:
      current_line.append(text)
      continue
    y0s = [t[-1][1] for t in current_line]
    y1s = [t[-1][3] for t in current_line]
    inter = np.minimum(y1s, text[-1][3]) - np.maximum(y0s, text[-1][1])
    inter = np.maximum(inter, 0)
    union = np.maximum(y1s, text[-1][3]) - np.minimum(y0s, text[-1][1])
    iou = inter / union
    if iou.mean() > iou_threshold:
      current_line.append(text)
    else:
      current_line = sorted(current_line, key=lambda x: (x[-1][0] + x[-1][2]) / 2)
      current_line.append(''.join([w[0] for w in current_line]))
      grouped.append(current_line)
      current_line = [text]
  current_line = sorted(current_line, key=lambda x: (x[-1][0] + x[-1][2]) / 2)
  current_line.append(''.join([w[0] for w in current_line]))
  grouped.append(current_line)
  return grouped


