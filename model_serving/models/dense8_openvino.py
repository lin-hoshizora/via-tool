from openvino.inference_engine import IENetwork
from .dense8_base import Dense8Base

class Dense8OpenVINO(Dense8Base):
  """
  OpenVINO wrapper for Dense8 inference
  """
  def __init__(self, model_path, ie_core, dev, logger, **kwargs):
    super().__init__(logger, height=64, **kwargs)
    if dev == 'CPU':
      net = ie_core.read_network(model=model_path+'.xml', weights=model_path+'.bin')
      self.exe = ie_core.load_network(net, dev, num_requests=kwargs.get('num_requests', 1))
    else:
      self.exe = ie_core.import_network(model_path, dev, num_requests=kwargs.get('num_requests', 1))
    self.nodes_in = list(self.exe.input_info.keys())
    self.nodes_out = list(self.exe.outputs.keys())
    assert len(self.nodes_in) == 1, f'{len(self.nodes_in)} inputs found, 1 expected'
    assert len(self.nodes_out) == 1, f'{len(self.nodes_out)} outputs found, 1 expected'
    self.node_image = self.nodes_in[0]
    self.node_logits = self.nodes_out[0]
    self.input_shape = self.exe.input_info[self.node_image].tensor_desc.dims

  def infer_sync(self, img, num_only=False):
    req = self.infer_async(img)
    req.wait()
    codes, probs, positions = self.get_result(req, num_only=num_only)
    return codes, probs, positions

  def infer_async(self, img):
    feed = self.preprocess(img)
    req = self.exe.start_async(self.exe.get_idle_request_id(), {self.node_image: feed})
    return req

  def get_result(self, req, num_only=False):
    req.wait()
    logits = req.output_blobs[self.node_logits].buffer
    codes, probs, positions = self.parse_result(logits, num_only)
    return codes, probs, positions
