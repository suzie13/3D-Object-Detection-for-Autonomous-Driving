from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


class COMPLEXYOLO(nn.Module):

    def __init__(self):
        super(COMPLEXYOLO, self).__init__()
