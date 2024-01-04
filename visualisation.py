from torchviz import make_dot
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import os

from torchvision.models import mobilenet_v2
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'



model = mobilenet_v2(weights=None)
model.eval()
dummy_input = torch.randn(1, 3, 32, 32)
# Forward pass to generate the graph
output = model(dummy_input)
make_dot(output, params=dict(model.named_parameters())).render("rnn_torchviz", format="png")