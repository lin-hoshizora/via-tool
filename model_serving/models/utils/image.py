import logging
import numpy as np
import cv2
try:
  import pyclipper
  from shapely.geometry import Polygon
except ModuleNotFoundError:
  print('Pyclipper and shapely are not installed.')

def resize_ar(img: np.ndarray, w: int, h: int, method: int =cv2.INTER_AREA):
  """
  Resize an image and keep aspect ratio
  """
  h0, w0 = img.shape[:2]
  img_pad = np.zeros((h, w, 3), dtype=np.uint8)
  scale = min(h / h0, w / w0)
  h1 = int(h0 * scale)
  w1 = int(w0 * scale)
  img_pad[:h1, :w1] = cv2.resize(img, (w1, h1), method)
  return img_pad, scale

def resize_h(img: np.ndarray, h: int, w: int, logger: logging.Logger):
  """
  Resize the image to a specified height,
  pad to ensure width is divisible by div
  :param img: original image
  :param h: target height
  :param w: target width
  :return: resized image with padding
  """
  h0, w0 = img.shape[:2]
  w1 = int(h / h0 * w0)
  img_resize = cv2.resize(img, (w1, h), cv2.INTER_AREA)
  img_pad = np.ones((h, w, 3), dtype=img_resize.dtype) * 200
  if img_resize.shape[1] > img_pad.shape[1]:
    logger.warn(f"Resized Image width {w1} > {w}, rightmost part cut off")
    img_resize = img_resize[:, :img_pad.shape[1], :]
  img_pad[:, :img_resize.shape[1], :] = img_resize
  return img_pad, img_resize

def nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float):
  """
  Apply NMS to bounding boxes
  :param boxes: boxes in an array with size [n*4]
  :param scores: score for each box
  :param iou_th: IOU threshold used in NMS
  :return: boxes after NMS
  """
  assert len(boxes) == len(scores)
  assert boxes.shape[1] == 4
  assert len(boxes.shape) == 2
  assert len(scores.shape) == 1

  areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0:
    keep.append(order[0])
    inter_x1 = np.maximum(boxes[order[0], 0], boxes[order[1:], 0])
    inter_y1 = np.maximum(boxes[order[0], 1], boxes[order[1:], 1])
    inter_x2 = np.minimum(boxes[order[0], 2], boxes[order[1:], 2])
    inter_y2 = np.minimum(boxes[order[0], 3], boxes[order[1:], 3])
    inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0)
    inter_area = inter_w * inter_h
    iou = inter_area / (areas[order[0]] + areas[order[1:]] - inter_area)

    rest_idx = np.where(iou <= iou_th)[0]
    order = order[rest_idx + 1]
  final_boxes = np.array(boxes[np.array(keep)])
  final_scores = np.array(scores[np.array(keep)])
  return final_boxes, final_scores

def get_rect(polygons: np.ndarray, min_wh_ratio: float = 0):
  polygons = polygons[:, :8]
  rects = []
  for polygon in polygons:
    pts = polygon.reshape(4, 2)
    x0 = pts[:, 0].min()
    x1 = pts[:, 0].max()
    y0 = pts[:2, 1].mean()
    y1 = pts[2:, 1].mean()
    if (x1 - x0) / (y1 - y0) > min_wh_ratio:
      rects.append([x0, y0, x1, y1])
  rects = np.array(rects)
  return rects

def get_chips(img: np.ndarray, boxes: np.ndarray):
  assert len(boxes.shape) == 2 and boxes.shape[1] == 4
  assert (boxes >= 0).all(), 'expect all coords to be non-negative'
  chips = np.array([img[b[1]:b[3], b[0]:b[2]]for b in boxes.astype(np.int32)])
  return chips

def match(img: np.ndarray, target: np.ndarray, method=cv2.TM_CCOEFF_NORMED, blur=5):
  img_blur = cv2.GaussianBlur(img, (blur, blur), 0)
  target_blur = cv2.GaussianBlur(target, (blur, blur), 0)
  res = cv2.matchTemplate(img_blur, target_blur, method)
  return res

def get_min_box(contour):
    bbox = cv2.minAreaRect(contour)
    pts = sorted(list(cv2.boxPoints(bbox)), key=lambda x: x[0])
    if pts[1][1] > pts[0][1]:
      idx1, idx4 = 0, 1
    else:
      idx1, idx4 = 1, 0
    if pts[3][1] > pts[2][1]:
      idx2, idx3 = 2, 3
    else:
      idx2, idx3 = 3, 2
    box = np.array(pts)[[idx1, idx2, idx3, idx4], ...]
    ssize = min(bbox[1])
    return box, ssize

def get_score(pred, contour):
    h, w = pred.shape
    c = contour.copy()
    xmin = np.clip(np.floor(c[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(c[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(c[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(c[:, 1].max()).astype(np.int), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    c[:, 0] -= xmin
    c[:, 1] -= ymin
    cv2.fillPoly(mask, c.reshape(1, -1, 2).astype(np.int32), 1)
    score = cv2.mean(pred[ymin: ymax + 1, xmin: xmax + 1], mask)[0]
    return score

def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def dist(pt1: np.array, pt2: np.array):
  """Calculate the distance between 2 points
  """
  dist = np.sqrt(np.square(np.abs(pt1 - pt2)).sum())
  return dist

def calc_angle(lines):
    angles = []
    lines = lines.reshape(-1, 8)
    for line in lines:
      hori_len = dist(line[:2], line[2:4])
      vert_len = dist(line[2:4], line[4:6])
      if hori_len > vert_len * 2:
        left = (line[:2] + line[6:8]) / 2
        right = (line[2:4] + line[4:6]) / 2
        dx, dy = right - left
        angle = np.arctan(dy / dx) / np.pi * 180
        angles.append(angle)
    angle = np.mean(angles)
    return angle

def rotate_lines(lines, angle, w, h):
    coords = lines[:, :8]
    coords = coords.reshape(-1, 2)
    coords = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    m = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    coords = np.dot(m, coords.transpose()).transpose()
    return coords
