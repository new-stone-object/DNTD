"""
the package store the loss that user defined
"""

import torch.nn.functional as F
import torch
from config import Configuration


def structure_weighted_binary_cross_entropy_with_logits(input, target:torch.Tensor):
    target_pad = F.pad(target,[10,10,10,10],mode='circular')
    weit = torch.abs(F.avg_pool2d(target_pad, kernel_size=21, stride=1, padding=0)-target)
    b,c,h,w = weit.shape
    weit = (weit-weit.view(b,c,-1).min(dim=-1,keepdim=True)[0].unsqueeze(-1)) / (1e-6+weit.view(b,c,-1).max(dim=-1,keepdim=True)[0].unsqueeze(-1)-weit.view(b,c,-1).min(dim=-1,keepdim=True)[0].unsqueeze(-1))
    dx = F.conv2d(F.pad(target, [1, 1, 0, 0], mode='reflect'),
                  torch.FloatTensor([-0.5, 0, 0.5]).view(1, 1, 1, 3).to(target.device), stride=1, padding=0)
    dy = F.conv2d(F.pad(target, [0, 0, 1, 1], mode='reflect'),
                  torch.FloatTensor([-0.5, 0, 0.5]).view(1, 1, 3, 1).to(target.device), stride=1, padding=0)
    torch.abs_(dx)
    torch.abs_(dy)
    edge_info = (dx + dy) > 0.4
    weit[edge_info] = 0.0
    weit = 1 + Configuration.instance().S_LOSS_GAMA * weit
    wbce  = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    wbce  = (weit*wbce)
    return wbce.sum()


if __name__ == '__main__':
    import sys
    sys.argv.append('-d')
    sys.argv.append('SOD')
    sys.argv.append('-save')
    sys.argv.append('test')
    a = torch.zeros(1,1,16,16).float()
    structure_weighted_binary_cross_entropy_with_logits(a,a)