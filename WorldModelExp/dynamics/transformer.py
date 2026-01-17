import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.blocks_tr import Transformer, TransformerEncoder

class 