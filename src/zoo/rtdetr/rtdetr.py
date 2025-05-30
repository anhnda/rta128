"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register
import time

__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.enc_time = 0
        self.dec_time = 0        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        start_t = time.time()    
        x = self.backbone(x)
        x = self.encoder(x)  
        self.enc_time += time.time() - start_t
        start_t = time.time()     
        x = self.decoder(x, targets)
        self.dec_time += time.time() - start_t
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
