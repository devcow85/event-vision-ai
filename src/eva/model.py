import pickle

import torch.nn as nn

import eva.loss as f
import eva.layers as L
from eva.utils import _convert_state_dict_to_numpy, _parse_extra_repr, _tensor_to_numpy

def conv_pool_block(in_channels, out_channels, kernel_size, padding, pooling_size, pooling_stride, tsslbp_config):
    return nn.Sequential(
        L.SNNConv3d(in_channels, out_channels, kernel_size, padding=padding, **tsslbp_config),
        L.SNNSumPooling(pooling_size, pooling_stride)
    )

# Base Network Class
class BaseNet(nn.Module):
    def __init__(self, tau_m, tau_s, n_steps):
        super(BaseNet, self).__init__()
        self.tsslbp_config = {
            'use_tsslbp': True,
            'tau_m': tau_m,
            'tau_s': tau_s,
            'n_steps': n_steps
        }
        self.layers = nn.ModuleList()  # A single ModuleList to store all layers (conv and fc)
        
        self.trace_dict = {}

    # Helper method to create layers dynamically
    def _make_layers(self, layer_configs):
        layers = []
        for layer_class, layer_params in layer_configs:
            layer = layer_class(**layer_params)
            layers.append(layer)
        return layers

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # Assuming the permutation is common across models

        for layer in self.layers:
            x = layer(x)

        return x
    
    def _hook_fn(self, name):
        
        def hook(module, inputs, outputs):
            self.trace_dict[name] = {
                "module_type": type(module).__name__,
                "inputs": tuple(_tensor_to_numpy(inp) for inp in inputs),
                "outputs": _tensor_to_numpy(outputs),
                "extra_repr": _parse_extra_repr(module.extra_repr()),
                "state_dict": _convert_state_dict_to_numpy(module.state_dict())
                }
        return hook
        
    def register_hook(self):
        for name, module in self.named_modules():
            # Check if the module has no submodules (leaf module)
            if len(list(module.children())) == 0:
        
                print(name)
                module.register_forward_hook(self._hook_fn(name))

    def weight_clipper(self):
        for name, module in self.named_modules():
            # Check if the module has no submodules (leaf module)
            if len(list(module.children())) == 0:
                if hasattr(module, 'weight_clipper'):
                    module.weight_clipper()
                    
    def save_trace_dict(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.trace_dict, f)

                
class NCARSNet(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps):
        super(NCARSNet, self).__init__(tau_m, tau_s, n_steps)
        
        # Configuration of layers for NCARSNet
        layer_configs = [
            (conv_pool_block, {'in_channels': 1, 'out_channels': 15, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 15, 'out_channels': 40, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 40, 'out_channels': 80, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 80, 'out_channels': 160, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 160, 'out_channels': 320, 'kernel_size': 3, 'padding': 1, 'pooling_size': 4, 'pooling_stride': 4, 'tsslbp_config': self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 320, 'out_features': 64, **self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 64, 'out_features': 2, **self.tsslbp_config}),
        ]

        # Create the layers dynamically and store them in self.layers
        self.layers = nn.ModuleList(self._make_layers(layer_configs))
                
                
                
                
                
# class DVSGestureNet(nn.Module):
#     def __init__(self, tau_m, tau_s, n_steps):
#         super(DVSGestureNet, self).__init__()
#         tsslbp_config = {'use_tsslbp':True, 
#                         'tau_m':tau_m,
#                         'tau_s':tau_s,
        
#                         'n_steps':n_steps}
        
#         self.block1 = conv_pool_block(2, 15, 5, 2, 2, 2, tsslbp_config)
#         self.block2 = conv_pool_block(15, 40, 5, 2, 2, 2, tsslbp_config)
#         self.block3 = conv_pool_block(40, 80, 3, 1, 2, 2, tsslbp_config)
#         self.block4 = conv_pool_block(80, 160, 3, 1, 2, 2, tsslbp_config)
#         self.block5 = conv_pool_block(160, 320, 3, 1, 2, 2, tsslbp_config)
        
#         self.fc1 = L.SNNLinear(5120, 512, **tsslbp_config)
#         self.fc2 = L.SNNLinear(512, 11, **tsslbp_config)
        
#     def forward(self, x):
#         x = x.permute(0,2,3,4,1)
        
#         y = self.block1(x)
#         y = self.block2(y)
#         y = self.block3(y)
#         y = self.block4(y)
#         y = self.block5(y)
        
#         y = self.fc1(y)
#         y = self.fc2(y)
        
#         return y

#     def weight_clipper(self):
#         for name, module in self.named_children():
#             if hasattr(module, 'weight_clipper'):
#                 module.weight_clipper()
                
# class NMNISTNet(nn.Module):
#     def __init__(self, tau_m, tau_s, n_steps):
#         super(NMNISTNet, self).__init__()
#         tsslbp_config = {'use_tsslbp':True, 
#                         'tau_m':tau_m,
#                         'tau_s':tau_s,
        
#                         'n_steps':n_steps}
        
#         self.block1 = conv_pool_block(2, 12, 5, 1, 2, 2, tsslbp_config)
#         self.block2 = conv_pool_block(12, 64, 5, 0, 2, 2, tsslbp_config)
#         self.fc = L.SNNLinear(2304, 10, **tsslbp_config)
    
#     def forward(self, x):
#         x = x.permute(0,2,3,4,1)
        
#         y = self.block2(self.block1(x))
#         return self.fc(y)
    
# class NCARSNet(nn.Module):
#     def __init__(self, tau_m, tau_s, n_steps):
#         super(NCARSNet, self).__init__()
#         tsslbp_config = {'use_tsslbp':True, 
#                         'tau_m':tau_m,
#                         'tau_s':tau_s,
        
#                         'n_steps':n_steps}
        
#         self.block1 = conv_pool_block(1, 15, 5, 2, 2, 2, tsslbp_config)
#         self.block2 = conv_pool_block(15, 40, 5, 2, 2, 2, tsslbp_config)
#         self.block3 = conv_pool_block(40, 80, 3, 1, 2, 2, tsslbp_config)
#         self.block4 = conv_pool_block(80, 160, 3, 1, 2, 2, tsslbp_config)
#         self.block5 = conv_pool_block(160, 320, 3, 1, 4, 4, tsslbp_config)
        
#         self.fc1 = L.SNNLinear(320, 64, **tsslbp_config)
#         self.fc2 = L.SNNLinear(64, 2, **tsslbp_config)
        
#     def forward(self, x):
#         x = x.permute(0,2,3,4,1)
        
#         y = self.block1(x)
#         y = self.block2(y)
#         y = self.block3(y)
#         y = self.block4(y)
#         y = self.block5(y)
        
#         y = self.fc1(y)
#         y = self.fc2(y)
        
#         return y