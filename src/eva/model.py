"""
model.py
history:
 - 23.04.17 refactoring
"""

import torch.nn as nn

import eva.loss as f
import eva.layers as L


def conv_pool_block(in_channels, out_channels, kernel_size, padding, pooling_size, pooling_stride, tsslbp_config):
    return nn.Sequential(
        L.SNNConv3d(in_channels, out_channels, kernel_size, padding=padding, **tsslbp_config),
        L.SNNSumPooling(pooling_size, pooling_stride)
    )

class DVSGestureNet(nn.Module):
    def __init__(self, tau_m, tau_s, n_steps):
        super(DVSGestureNet, self).__init__()
        tsslbp_config = {'use_tsslbp':True, 
                        'tau_m':tau_m,
                        'tau_s':tau_s,
        
                        'n_steps':n_steps}
        
        self.block1 = conv_pool_block(2, 15, 5, 2, 2, 2, tsslbp_config)
        self.block2 = conv_pool_block(15, 40, 5, 2, 2, 2, tsslbp_config)
        self.block3 = conv_pool_block(40, 80, 3, 1, 2, 2, tsslbp_config)
        self.block4 = conv_pool_block(80, 160, 3, 1, 2, 2, tsslbp_config)
        self.block5 = conv_pool_block(160, 320, 3, 1, 2, 2, tsslbp_config)
        
        self.fc1 = L.SNNLinear(5120, 512, **tsslbp_config)
        self.fc2 = L.SNNLinear(512, 11, **tsslbp_config)
        
    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        
        y = self.fc1(y)
        y = self.fc2(y)
        
        return y

    def weight_clipper(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight_clipper'):
                # `weight_clipper` 메서드가 있는 경우, 해당 메서드를 실행
                module.weight_clipper()
                
class NMNISTNet(nn.Module):
    def __init__(self, tau_m, tau_s, n_steps):
        super(NMNISTNet, self).__init__()
        tsslbp_config = {'use_tsslbp':True, 
                        'tau_m':tau_m,
                        'tau_s':tau_s,
        
                        'n_steps':n_steps}
        
        self.block1 = conv_pool_block(2, 12, 5, 1, 2, 2, tsslbp_config)
        self.block2 = conv_pool_block(12, 64, 5, 0, 2, 2, tsslbp_config)
        self.fc = L.SNNLinear(2304, 10, **tsslbp_config)
    
    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        
        y = self.block2(self.block1(x))
        return self.fc(y)
    
class NCARSNet(nn.Module):
    def __init__(self, tau_m, tau_s, n_steps):
        super(NCARSNet, self).__init__()
        tsslbp_config = {'use_tsslbp':True, 
                        'tau_m':tau_m,
                        'tau_s':tau_s,
        
                        'n_steps':n_steps}
        
        self.block1 = conv_pool_block(1, 15, 5, 2, 2, 2, tsslbp_config)
        self.block2 = conv_pool_block(15, 40, 5, 2, 2, 2, tsslbp_config)
        self.block3 = conv_pool_block(40, 80, 3, 1, 2, 2, tsslbp_config)
        self.block4 = conv_pool_block(80, 160, 3, 1, 2, 2, tsslbp_config)
        self.block5 = conv_pool_block(160, 320, 3, 1, 4, 4, tsslbp_config)
        
        self.fc1 = L.SNNLinear(320, 64, **tsslbp_config)
        self.fc2 = L.SNNLinear(64, 2, **tsslbp_config)
        
    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        
        y = self.fc1(y)
        y = self.fc2(y)
        
        return y