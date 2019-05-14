# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import collections
import tensorflow as tf

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

