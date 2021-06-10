import numpy as np
import cv2
from .utils.image import resize_ar, get_min_box, get_score, unclip, calc_angle, rotate_lines

class DBNet:
  def __init__(self,
               logger,
               threshold: float = 0.3,
               box_th: float = 0.5,
               max_candidates: int = 1000,
               unclip_ratio: float = 2.,
               min_size: int = 3,
               **kwargs):
    self.logger = logger
    self.threshold = threshold
    self.box_th = box_th
    self.max_candidates = max_candidates
    self.unclip_ratio = unclip_ratio
    self.min_size = min_size

  def preprocess(self, img: np.ndarray, **kwargs):
    img, self.scale = resize_ar(img, self.input_w, self.input_h)
    self.img = img.copy()[np.newaxis, ...].astype(np.float32)
    return img

  def parse_result(self, result: np.ndarray):
    result = result[0, ..., 0]
    res_exp = np.exp(result)
    result = res_exp / (res_exp + 1)
    mask = result > self.threshold
    h, w = mask.shape
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    n_contours = min(len(contours), self.max_candidates)
    boxes = []
    scores = []
    rects = []
    for idx, contour in enumerate(contours[:n_contours]):
      c = contour.squeeze(1)
      pts, sside = get_min_box(c)
      if sside < self.min_size:
        continue
      score = get_score(result, c)
      if self.box_th > score:
        continue
      c = unclip(pts, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
      pts, sside = get_min_box(c)
      if sside < self.min_size + 2:
        continue
      pts[:, 0] = np.clip(np.round(pts[:, 0]), 0, w)
      pts[:, 1] = np.clip(np.round(pts[:, 1]), 0, h)
      boxes.append(pts.astype(np.float32))
      scores.append(score)
    boxes = np.array(boxes)
    scores = np.array(scores)
    boxes /= self.scale
    angle = calc_angle(boxes)
    if np.abs(angle) > 0.2: boxes = rotate_lines(boxes, w=w / self.scale, h=h / self.scale, angle=angle)
    return boxes, angle
