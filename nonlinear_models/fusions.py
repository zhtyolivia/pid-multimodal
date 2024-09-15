# The code in this script is modified from:
# # https://github.com/pliang279/PID/blob/main/fusions/common_fusions.py

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self):
        """Initialize Concat Module."""
        super(Concat, self).__init__()

    def forward(self, modalities):
        """
        Apply concatenation to input 
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)

class TensorFusion(nn.Module):
    """
    Implementation of TensorFusion Networks.
    
    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """
    def __init__(self, concat_1=True):
        """Instantiates TensorFusion Network Module."""
        super().__init__()
        self.concat_1 = concat_1

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        if self.concat_1:
            m = torch.cat((Variable(torch.ones(
                *nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0), dim=-1)
        else:
            m = torch.cat([mod0], dim=-1)

        for mod in modalities[1:]:
            if self.concat_1:
                mod = torch.cat((Variable(torch.ones(
                    *nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod), dim=-1)
            else:
                mod = torch.cat([mod], dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])
        return m