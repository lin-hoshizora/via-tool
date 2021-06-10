import time
import grpc
import yaml
import numpy as np
import cv2
from . import model_serving_pb2
from . import model_serving_pb2_grpc
from .utils import get_dense_key, get_layout, load_conf

class ModelServerClient:
  """A model server client to handle gRPC communication"""
  def __init__(self, docker_manager, logger):
    self.config = load_conf('config/model_server_config.yaml')
    self.err = load_conf('config/model_server_err.yaml')
    self.logger = logger
    self.ip_pool = [self.config['grpc']['ip'], *self.config['grpc']['remote']]
    self.docker_manager = docker_manager
    self.t_restart = 0
    self.logger.info(f'Restart cooldown of model server: {self.config["grpc"]["restart_cooldown"]}s')
    self.docker_manager.run_if_not_yet(cooldown=0, **self.config['docker'])

  def _check_infer(self, stub, sess_id):
    """Test model inference to check NSC2 status
    """
    timeout = self.config['grpc']['timeout']['check']
    for trial_idx in range(self.config['grpc']['test_trials']):
      try:
        res = stub.Check(model_serving_pb2.CheckRequest(sess_id=sess_id), timeout=timeout)
        res_json = {"Result": res.status}
        break
      except grpc._channel._InactiveRpcError as e:
        self.logger.warning(f'InferCheck retrials after timeout: {trial_idx + 1}')
        time.sleep(self.config['grpc']['test_cooldown'])
        res_json = {"Result": "NG"}
    return res_json

  def _make_dense_req(self, sess_id, img, key, num_only):
    """Prepare a gRPC request for text recognition
    """
    img_pb = model_serving_pb2.Image(data=img.tobytes(), h=img.shape[0], w=img.shape[1], c=img.shape[2])
    if key is None: key = get_dense_key(img)
    req = model_serving_pb2.DenseRequest(sess_id=sess_id, img=img_pb, key=key, num_only=num_only)
    return req

  def _sleep_if_necessary(self, infer_idx, trials):
    """Sleep when server is inactive and one more trial is allowed"""
    if infer_idx < trials - 1:
      self.logger.warning(f'Inactive inference server, retry after {self.config["grpc"]["infer_cooldown"]}s')
      time.sleep(self.config['grpc']['infer_cooldown'])

  def _dense_infer(self, stub, sess_id, img, key, num_only, trials):
    """Text recognition inference RPC
    """
    timeout = self.config['grpc']['timeout']['dense']
    req_options = {
      "sess_id": sess_id,
      "key": key,
      "num_only": num_only
    }
    for infer_idx in range(trials):
      try:
        if isinstance(img, str):
          req = model_serving_pb2.DensePathRequest(path=img, **req_options)
          res = stub.DensePathInferSync(req, timeout=timeout)
        else:
          req = self._make_dense_req(img=img, **req_options)
          res = stub.DenseInferSync(req, timeout=timeout)
        code = np.frombuffer(res.code, np.float32).copy()
        code = code.astype(np.int64)
        prob = np.frombuffer(res.prob, np.float32).copy()
        pos = np.frombuffer(res.position, np.float32).copy()
        res_json = {'code': code, 'prob': prob, 'position': pos}
      except grpc._channel._InactiveRpcError as e:
        res_json = self.err['ocr_err']
        self._sleep_if_necessary(infer_idx, trials)
    return res_json

  def _dense_req_iterator(self, sess_id, imgs, keys, num_onlys):
    """Make a text recognition request generator for stream"""
    def req_gen():
      for i, img in enumerate(imgs):
        yield self._make_dense_req(sess_id, img, keys[i], num_onlys[i])
    iterator = req_gen()
    return iterator

  def _clean_dense_res(self, results, n):
    """Clean results from gRPC server"""
    results_clean = [r[r!=-1] for r in results.reshape((n, -1))]
    return results_clean

  def _dense_batch_infer(self, stub, sess_id, imgs, keys, num_onlys, trials=1):
    """Text recognition batch inference RPC

    This ONLY helps with the inference speed when NCS2 is used.
    Since NCS2 uses USB interface to transfer data, it will be much
    more efficient to keep the data transfer going when waiting for
    a model output.
    """
    timeout = self.config['grpc']['timeout']['dense_batch']
    for infer_idx in range(trials):
      try:
        res = stub.DenseBatchInferSync(self._dense_req_iterator(sess_id, imgs, keys, num_onlys))
        codes = np.frombuffer(res.codes, np.float32).copy()
        codes = self._clean_dense_res(codes, n=len(imgs))
        codes = [c.astype(np.int64) for c in codes]
        probs = np.frombuffer(res.probs, np.float32).copy()
        probs = self._clean_dense_res(probs, n=len(imgs))
        positions = np.frombuffer(res.positions, np.float32).copy()
        positions = self._clean_dense_res(positions, n=len(imgs))
        res_json = {'codes': codes, 'probs': probs, 'positions': positions}
      except grpc._channel._InactiveRpcError as e:
        res_json = self.err['ocr_err']
        self._sleep_if_necessary(infer_idx, trials)
    return res_json

  def _det_infer(self, stub, sess_id, img, layout, suppress_lines, trials):
    """Text detection inference RPC

    Ars:
      layout: Calculated by `get_layout` if set to `None` and img is an `np.ndarray`.
        Note that it can be set to any supported value which is not necesarrily consistent
        with the output of `get_layout`
    """
    timeout = self.config['grpc']['timeout']['det']
    req_options = {
      "sess_id": sess_id,
      "layout": layout,
      "suppress_lines": suppress_lines
    }
    for infer_idx in range(trials):
      try:
        if isinstance(img, str):
          req = model_serving_pb2.DetPathRequest(path=img, **req_options)
          res = stub.DetPathInferSync(req, timeout=timeout)
        else:
          img_pb = model_serving_pb2.Image(data=img.tobytes(),
                                           h=img.shape[0], w=img.shape[1], c=img.shape[2])
          if layout is None: layout = get_layout(img)
          req = model_serving_pb2.DetRequest(img=img_pb, **req_options)
          res = stub.DetInferSync(req, timeout=timeout)
        lines = np.frombuffer(res.lines, np.float32).copy()
        lines = lines.reshape((-1, 8))
        res_json = {'lines': lines.copy(), 'angle': res.angle}
      except grpc._channel._InactiveRpcError as e:
        res_json = self.err['ocr_err']
        if infer_idx < trials - 1:
          self.logger.warning(f'Inactive inference server,'
                              f'retry after {self.config["grpc"]["infer_cooldown"]}s')
          time.sleep(self.config['grpc']['infer_cooldown'])
    return res_json

  def in_cooldown(self):
    """Check if the model server is in cooldown (initialization phase)"""
    if time.time() - self.t_restart < self.config['grpc']['restart_cooldown']:
      return True
    return False

  def restart(self):
    """Restart model_server docker container"""
    self.docker_manager.stop(self.config['docker']['image'])
    self.docker_manager.run_if_not_yet(cooldown=0, **self.config['docker'])
    self.t_restart = time.time()


  def restart_if_local(self, ip):
    """Restart local model_server"""
    if ip != self.ip_pool[0]: return
    if not self.in_cooldown():
      self.logger.info('Restart local model_server')
      self.restart()
    else:
      self.logger.info('Just restarted local model_server, ignored restart signal.')

  def infer_sync(self, sess_id, network, img, key=None, num_only=None, layout=None, suppress_lines=None,
                 check_local=True):
    """Run inference of text detection or recognition
    """
    # gRPC options
    port = self.config['grpc']['port']
    options = None
    max_msg_len = self.config['grpc'].get('max_msg_len', None)
    if max_msg_len is not None:
      options = [
        ('grpc.max_send_message_length', int(max_msg_len * 1024 * 1024)),
        ('grpc.max_receive_message_length', int(max_msg_len * 1024 * 1024)),
      ]

    for ip in self.ip_pool:
      # Do nothing and return an error when model_server is cooling down
      if self.in_cooldown() and ip == self.ip_pool[0]:
        self.logger.info("Just restarted local model_server, skip inference on it")
        if network == 'Check': res_json = {'Result': 'NG'}
        else: res_json = self.err['ocr_err']
        continue

      with grpc.insecure_channel(f'{ip}:{port}', options=options) as channel:
        self.logger.info(f'{network} Infer at {ip}:{port}')
        stub = model_serving_pb2_grpc.ModelServerStub(channel)
        if network == 'Check':
          # inference check
          res_json = self._check_infer(stub, sess_id)
        else:
          if (ip != self.ip_pool[0] or check_local):
            # run inference check if configured so or not using local server
            test_json = self._check_infer(stub, sess_id)
            if test_json.get('Result', 'NG') == 'NG':
              self.logger.error(f'Infer check failed @ {ip}')
              res_json = self.err['ocr_err']
              self.restart_if_local(ip)
              continue
          if network == 'Dense':
            # text recognition
            res_json = self._dense_infer(stub, sess_id, img, key, num_only, trials=1)
          elif network == 'Det':
            # text detection
            res_json = self._det_infer(stub, sess_id, img, layout, suppress_lines, trials=self.config['grpc']['infer_trials'])
          else:
            res_json = {
              "ErrCode": "E000",
              "ErrMsg": f"{network} is not implemented in model_serving."
            }
            return res_json
        if 'ErrCode' in res_json or res_json == {"Result": "NG"}:
          # restart model_server if any inference failed
          self.logger.error(f'{res_json.get("ErrMsg", "Infer check failed")} @ {ip}')
          self.restart_if_local(ip)
          if 'Result' in res_json:
            res_json = self.err['ocr_err']
        else:
          return res_json
    return res_json

  def infer_batch_sync(self, sess_id, network, imgs, num_onlys=None, check_local=True):
    """Run batch inference for text recognition"""
    if network != 'Dense':
      # only text recognition supports batch inference
      res_json = {
        "ErrCode": "E000",
        "ErrMsg": f"{network} is not implemented for batch inference."
      }
      return res_json

    # broadcast num_only option
    if not isinstance(num_onlys, list): num_onlys = [num_onlys] * len(imgs)

    # gRPC config
    port = self.config['grpc']['port']
    options = None
    max_msg_len = self.config['grpc'].get('max_msg_len', None)
    if max_msg_len is not None:
      options = [
        ('grpc.max_send_message_length', int(max_msg_len * 1024 * 1024)),
        ('grpc.max_receive_message_length', int(max_msg_len * 1024 * 1024)),
      ]

    for ip in self.ip_pool:
      # do nothing if model_server is cooling down
      if self.in_cooldown() and ip == self.ip_pool[0]:
        self.logger.info("Just restarted local model_server, skip inference on it")
        res_json = self.err['ocr_err']
        continue
      with grpc.insecure_channel(f'{ip}:{port}', options=options) as channel:
        self.logger.info(f'{network} Batch Infer at {ip}:{port}')
        stub = model_serving_pb2_grpc.ModelServerStub(channel)
        if (ip != self.ip_pool[0] or check_local):
          test_json = self._check_infer(stub, sess_id)
          if test_json.get('Result', 'NG') == 'NG':
            self.logger.error(f'Infer check failed @ {ip}')
            res_json = self.err['ocr_err']
            self.restart_if_local(ip)
            continue
        res_json = self._dense_batch_infer(stub, sess_id, imgs, [None]*len(imgs), num_onlys)
      if 'ErrCode' in res_json:
        self.logger.error(f'{res_json.get("ErrMsg", "Batch infer failed")} @ {ip}')
        self.restart_if_local(ip)
      else:
        return res_json
    return res_json
