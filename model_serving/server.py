from concurrent import futures
import logging
from pathlib import Path
import grpc
import yaml
import numpy as np
import cv2
from openvino.inference_engine import IECore

from . import models
from . import model_serving_pb2
from . import model_serving_pb2_grpc
from .utils import get_logger, decode_img, pad1d, load_conf


class ModelServer(model_serving_pb2_grpc.ModelServerServicer):
  def __init__(self, config, ie_core):
    super().__init__()
    self.err = load_conf('config/model_server_err.yaml')
    self.config = config
    self.logger = get_logger(config['logger'])
    self.logger.info('Starting to init model server...')
    self.ie_core = ie_core
    self.model_folder = Path(config['model_folder'])
    if not self.model_folder.exists():
      raise ValueError(f"Model folder {str(self.model_folder)} does not exists. Check your model_server_config.yaml.")
    self._load_det()
    self._load_recog()
    self._load_test()

  def _det_params(self):
    model = self.config["grpc"]["det_model"]
    model_tag = model.lower().replace("openvino", "").replace("armnn", "")
    dev = self.config[model_tag]['dev']
    subfolder = self.config[model_tag].get('precision', None)
    if subfolder is None:
      if dev == 'MYRIAD': subfolder = 'FP16'
      elif dev == 'CPU': subfolder = 'FP32'
      else: raise NotImplementedError(f"CTPNP is not implemented for {dev}")
    path_portrait = str(self.model_folder / subfolder / self.config[model_tag]['model_path']['portrait'])
    path_landscape = str(self.model_folder / subfolder / self.config[model_tag]['model_path']['landscape'])
    batch_size = self.config[model_tag]['usb_batch_size']
    return model, path_portrait, path_landscape, dev, batch_size

  def _dense8_params(self):
    model = self.config["grpc"]["recog_model"]
    dev = self.config['dense8']['dev']
    subfolder = self.config['dense8'].get('precision', None)
    if subfolder is None:
      if dev == 'MYRIAD':
        subfolder = 'FP16'
      elif dev == 'CPU':
        subfolder = 'FP32'
      else:
        raise NotImplementedError(f"Dense8 is not implemented for {dev}")
    paths = self.config["dense8"]["model_list"]
    paths = {k : str(self.model_folder / subfolder / v) for k, v in paths.items()}
    batch_size = self.config['dense8']['usb_batch_size']
    return model, paths, dev, batch_size

  def _load_det(self):
    self.det = {}
    model, path_portrait, path_landscape, dev, batch_size = self._det_params()
    options = {
      "ie_core": self.ie_core,
      "dev": dev,
      "logger": self.logger,
      "num_requests": batch_size,
    }
    self.det['portrait'] = getattr(models, model)(path_portrait, **options)
    self.det['landscape'] = getattr(models, model)(path_landscape, **options)
    self.logger.info('Detector models loaded.')

  def _load_recog(self):
    model, paths, dev, batch_size = self._dense8_params()
    options = {
      "ie_core": self.ie_core,
      "dev": dev,
      "logger": self.logger,
      "num_requests": batch_size,
    }
    self.recog = {k: getattr(models, model)(p, **options) for k, p in paths.items()}
    self.logger.info('Recognizer models loaded.')

  def _load_test(self):
    test_config = self.config['check_model']
    model_folder = Path(test_config['model_folder'])
    stem = str(model_folder / test_config['model'])
    data_folder = Path(test_config['data_folder'])
    input_paths = [str(data_folder / fname ) for fname in test_config['input']]
    output_paths = [str(data_folder / fname) for fname in test_config['ref_output']]
    if test_config['dev'] == 'CPU':
      net = self.ie_core.read_network(model=stem+'.xml', weights=stem+'.bin')
      self.test_model = self.ie_core.load_network(net, 'CPU', num_requests=test_config['usb_batch_size'])
    else:
      self.test_model = self.ie_core.import_network(stem, test_config['dev'], num_requests=test_config['usb_batch_size'])
    self.test_in = [np.load(p) for p in input_paths]
    self.test_ref_out = [np.load(p) for p in output_paths]
    self.test_in_name = list(self.test_model.input_info.keys())[0]
    self.test_out_name = list(self.test_model.outputs.keys())[0]
    self.logger.info('Test model and data loaded.')

  def Check(self, request, context):
    sess_id = request.sess_id
    self.logger.info('Tests started')
    reqs = []
    for idx, (test_in, test_ref_out) in enumerate(zip(self.test_in, self.test_ref_out)):
      reqs.append(self.test_model.start_async(self.test_model.get_idle_request_id(), {self.test_in_name: test_in}))

    for idx, (req, test_ref_out) in enumerate(zip(reqs, self.test_ref_out)):
      req.wait()
      test_out = req.output_blobs[self.test_out_name].buffer
      if np.allclose(test_out, test_ref_out):
        status = 'OK'
        self.logger.info(f'Passed test{idx}')
      else:
        status = 'NG'
        self.logger.error(f'Failed test{idx}')
        break
    res = model_serving_pb2.CheckResponse(sess_id=sess_id, status=status)
    self.logger.info(f'Tests done. Result: {status}')
    return res

  def DetInferSync(self, request, context):
    sess_id = request.sess_id
    self.logger.info(f'Started {request.layout} detection inference for sess {sess_id}')
    img = decode_img(request.img)
    if request.layout in self.det:
      lines, angle = self.det[request.layout].infer_sync(img, suppress_lines=request.suppress_lines)
    else:
      self.logger.error(f"Layout {request.layout} is not supported")
      lines = []
    lines = np.array(lines, dtype=np.float32)
    lines_bytes = np.ndarray.tobytes(lines)
    res = model_serving_pb2.DetResponse(sess_id=sess_id, lines=lines_bytes, angle=angle)
    self.logger.info(f'Finished detection inference for sess {sess_id}')
    return res

  def DenseInferSync(self, request, context):
    sess_id = request.sess_id
    img = decode_img(request.img)
    code, prob, position = self.recog[request.key].infer_sync(img, num_only=request.num_only)
    code_bytes = np.ndarray.tobytes(code.astype(np.float32))
    prob_bytes = np.ndarray.tobytes(prob.astype(np.float32))
    position_bytes = np.ndarray.tobytes(position.astype(np.float32))
    res = model_serving_pb2.DenseResponse(sess_id=sess_id, code=code_bytes, prob=prob_bytes, position=position_bytes)
    return res

  def DenseBatchInferSync(self, request_iterator, context):
    reqs = []
    results = []
    for idx, request in enumerate(request_iterator):
      sess_id = request.sess_id
      if idx == 0:
        self.logger.info(f'Started dense batch inference for sess {sess_id}')
      img = decode_img(request.img)
      reqs.append((self.recog[request.key].infer_async(img), request.key, request.num_only))
      if len(reqs) >= self.config['dense8']['usb_batch_size']:
        results += [self.recog[key].get_result(req, num_only) for req, key, num_only in reqs]
        reqs.clear()
    results += [self.recog[key].get_result(req, num_only) for req, key, num_only in reqs]
    max_len = max([res[0].shape[0] for res in results])
    codes = []
    probs = []
    positions = []
    for code, prob, position in results:
       codes.append(pad1d(code, length=max_len, constant=-1))
       probs.append(pad1d(prob, length=max_len, constant=-1))
       positions.append(pad1d(position, length=max_len, constant=-1))
    codes = np.array(codes).astype(np.float32)
    probs = np.array(probs).astype(np.float32)
    positions = np.array(positions).astype(np.float32)
    codes_bytes = np.ndarray.tobytes(codes)
    probs_bytes = np.ndarray.tobytes(probs)
    positions_bytes = np.ndarray.tobytes(positions)
    res = model_serving_pb2.DenseBatchResponse(sess_id=sess_id, n=codes.shape[0], codes=codes_bytes, probs=probs_bytes,
                                               positions=positions_bytes)
    self.logger.info(f'Finished dense batch inference for sess {sess_id}')
    return res

def serve():
  logging.basicConfig()
  with open('config/model_server_config.yaml') as f:
    config = yaml.safe_load(f)
  max_workers = config['grpc'].get('max_workers', 1)
  options = None
  max_msg_len = config['grpc'].get('max_msg_len', None)
  if max_msg_len is not None:
    options = [
      ('grpc.max_send_message_length', int(max_msg_len * 1024 * 1024)),
      ('grpc.max_receive_message_length', int(max_msg_len * 1024 * 1024)),
    ]
  max_concurrent_rpcs = config['grpc'].get('max_concurrent_rpcs', None)
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options,
                       maximum_concurrent_rpcs=max_concurrent_rpcs)
  model_servicer = ModelServer(config, IECore())
  model_serving_pb2_grpc.add_ModelServerServicer_to_server(model_servicer, server)
  ip = config['grpc'].get('ip', '127.0.0.1')
  port = config['grpc'].get('port', 50052)
  model_servicer.logger.info(f'Start server at {ip}:{port}')
  server.add_insecure_port(f'{ip}:{port}')
  server.start()
  server.wait_for_termination()

if __name__ == "__main__":
  serve()
