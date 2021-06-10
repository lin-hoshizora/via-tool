from openvino.inference_engine import IENetwork
import numpy as np
from .dbnet_base import DBNet


class DBNetOpenVINO(DBNet):
  """
  OpenVINO wrapper for DBNet inference
  """
  def __init__(self, model_path, ie_core, dev, logger, **kwargs):
    super().__init__(logger, **kwargs)
    if dev == "CPU":
      net = ie_core.read_network(model=model_path + ".xml", weights=model_path + ".bin")
      self.exe = ie_core.load_network(net, dev, num_requests=kwargs.get("num_requests", 1))
    else:
      if not model_path.endswith("_MYRIAD"): model_path += "_MYRIAD"
      self.exe = ie_core.import_network(model_path, dev, num_requests=kwargs.get("num_requests", 1))
    self.nodes_in = list(self.exe.input_info.keys())
    self.nodes_out = list(self.exe.outputs.keys())
    assert len(self.nodes_in) == 1, '{len(self.nodes_in) inputs found, 1 expected}'
    assert len(self.nodes_out) == 1, '{len(self.nodes_out) inputs found, 1 expected}'
    self.input_shape = self.exe.input_info[self.nodes_in[0]].tensor_desc.dims
    self.input_h = self.input_shape[1]
    self.input_w = self.input_shape[2]


  def infer_sync(self, img: np.ndarray, **kwargs):
    feed = self.preprocess(img)
    req = self.exe.start_async(0, {self.nodes_in[0]: feed})
    req.wait()
    res = req.output_blobs[self.nodes_out[0]].buffer
    boxes, angle = self.parse_result(res)
    return boxes, angle
