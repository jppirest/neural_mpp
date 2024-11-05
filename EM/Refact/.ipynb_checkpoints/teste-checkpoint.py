import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('/home/joao.pires/MPP/neural_mpp/EM/Refact/')
from models import NormalizingFlow
from sweep import HawkesSweep

hk = HawkesSweep([[1, 2, 3], [4, 5, 6]], 2)


print(hk.make_dict())