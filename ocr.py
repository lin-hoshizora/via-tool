from typing import List,Any
from model_serving.client import ModelServerClient
import numpy as np
import pickle
import json


class FakeDockerManger:
    def run_if_not_yet(*args, **kwargs):
        pass
    def stop(*args, **kwargs):
        pass
class FakeLogger:
    def info(*args, **kwargs):
        pass
    def error(*args, **kwargs):
        pass
logger=FakeLogger()
docker_manager = FakeDockerManger()
model_server = ModelServerClient(docker_manager=docker_manager, logger=logger)

id2_path='./id2char_std.pkl'
with open(id2_path, 'rb') as f:
    id2char = pickle.load(f)

def get_file_size(file_path: str) -> int:
    with open(file_path, 'rb') as f:
        f.seek(0,2)
        size = f.tell()
    return size




def get_chips(img: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
    """Gets images chips from an image based on given bounding boxes.

    Args:
    img: Image to extract chips from
    boxes: Bounding boxes

    Returns:
    A list of image chips, each of which is a numpy array.
    """
    assert len(boxes.shape) == 2 and boxes.shape[1] == 4
    assert (boxes >= 0).all(), 'expect all coords to be non-negative'
    chips = []
    for b in boxes.astype(int):
        x1 = min(max(b[0], 0), img.shape[1] - 2)
        x2 = max(b[2], x1 + 1)
        y1 = min(max(b[1], 0), img.shape[0] - 2)
        y2 = max(b[3], y1 + 1)
        chips.append(img[y1:y2, x1:x2])
    return chips
def get_rect(polygons: np.ndarray, min_wh_ratio: float = 0) -> np.ndarray:
    """Gets rectangles from polygons.

    Args:
    polygons: 2d numpy array with a shape of (n, 8), where n is number of
      polygons. If the 2nd dimension is larger than 8, only the first 8
      numbers will be used.
    min_wh_ratio: Minimum width-height ratio for a valid rectangle

    Returns:
    A 2d numpy array with a shape of (n, 4), where n is number of rectangles.
    """
    polygons = polygons[:, :8]
    rects = []
    for polygon in polygons:
        pts = polygon.reshape(4, 2)
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        #y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        y0, y1 = pts[:2, 1].mean(), pts[2:, 1].mean()
        if y1 - y0 < 8 or x1 - x0 < 8: continue
        if (x1 - x0) / (y1 - y0) > min_wh_ratio:
            rects.append([x0, y0, x1, y1])
    rects = np.array(rects)
    return rects
def group_lines(
    texts: List[List[Any]],
    iou_threshold: float = 0.4
) -> List[List[Any]]:
    """Groups texts with bounding boxes in lines.

    Args:
    texts: A list of OCR result, each element of which is also a list of
      [text, probability, position, bouding_box]
    iou_threshold: Threshold for IOU in vertical direction to determine if
      two bounding boxes belong to the same line.

    Returns:
    A list of lists of bouding boxes belonging to the same line.
    """
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

def ocr(img):
    if img.shape[0] > img.shape[1]:
        layout = 'portrait'
    else:
        layout = 'landscape'

    det_res = model_server.infer_sync(
        sess_id='123', network='Det',
        img=img,
        layout=layout,
        suppress_lines=False,
        check_local=False
    )
    det_res
    lines=det_res['lines']
#     print(len(lines))
    if len(lines) == 0:
        return False
    text_boxes = get_rect(lines, min_wh_ratio=0.5)
#     print(len(text_boxes))
#     print(text_boxes)
    chips = get_chips(img,abs(text_boxes))

    recog_res_dict = model_server.infer_batch_sync(
            sess_id='123',
            network='Dense',
            imgs=chips,
            num_onlys=[False]*len(text_boxes),
            check_local=False
        )
    pth=0.5
    recog_results = []
    for idx, (box, codes) in enumerate(zip(
            abs(text_boxes),
            recog_res_dict["codes"]
        )):
        probs, positions = (
          recog_res_dict["probs"][idx],
          recog_res_dict["positions"][idx],
        )
        if codes.size == 0:
            continue
        indices = probs > pth
        probs = probs[indices]
        positions = positions[indices]
        codes = codes[indices]
        text = "".join([id2char[c] for c in codes])
        if text:
            recog_results.append([text, probs, positions, box])

        # group text areas in lines
        texts = group_lines(recog_results)
    
    return texts
    

def read_base_json(json_path='base.json'):
    with open(json_path,'r') as f:
        bases = json.load(f)
    return bases