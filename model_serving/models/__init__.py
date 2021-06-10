try:
  from .ctpnp_openvino import CTPNP_OpenVINO
  from .dense8_openvino import Dense8OpenVINO
  from .dbnet_openvino import DBNetOpenVINO
except ModuleNotFoundError:
  print('OpenVINO not installed')
