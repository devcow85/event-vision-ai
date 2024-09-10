from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Union

import yaml
import torch

class Parse(object):
    """
    This class reads yaml parameter file and allows dictionary like access to the members.
    """

    def __init__(self, path):
        self.parameters = yaml.safe_load(path)
        # with open(path, "r", encoding="UTF8") as file:
        #     self.parameters = yaml.safe_load(file)

    def __getitem__(self, key):
        return self.parameters[key]

    def save(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self.parameters, f)


@dataclass
class Networks:
    """
    Network key dataclass
    """

    lr: float
    epochs: int
    batch_size: int
    n_steps: int
    dt: int
    version: str
    input_device: str
    resize_resolution: list
    rule: str
    loss: str
    desired_count: int
    undesired_count: int
    tau_m: int
    tau_s: int
    snn_model: str
    dataset: str = None
    seed: int = -1
    create_date: datetime = field(default_factory=datetime.now)
    device: str = None
    syn_a = torch.Tensor
    dtype = torch.float32

    def __post_init__(self):


        print(str(self.device) + " is available")

        self.syn_a = torch.zeros(
            (1, 1, 1, 1, self.n_steps), dtype=self.dtype, device=self.device
        )
        self.syn_a[..., 0] = 1
        for t in range(self.n_steps - 1):
            self.syn_a[..., t + 1] = (
                self.syn_a[..., t] - self.syn_a[..., t] / self.tau_s
            )
        self.syn_a /= self.tau_s



@dataclass
class ConvParam:
    """
    convolution3d layer config
    """

    name: str
    type: str
    in_channels: int
    out_channels: int
    threshold: float
    kernel_size: Union[int, tuple]
    groups: int = 1
    padding: Union[int, tuple] = 0
    stride: Union[int, tuple] = 1
    dilation: Union[int, tuple] = 1
    weight_scale: float = 1.0

    def __post_init__(self):
        # kernel
        if type(self.kernel_size) == int:
            self.kernel_size = (self.kernel_size, self.kernel_size, 1)
        elif len(self.kernel_size) == 2:
            self.kernel_size = (self.kernel_size[0], self.kernel_size[1], 1)
        else:
            raise Exception(
                "kernelSize can only be of 1 or 2 dimension. It was: {}".format(
                    self.kernel_size.shape
                )
            )

        # stride
        if type(self.stride) == int:
            self.stride = (self.stride, self.stride, 1)
        elif len(self.stride) == 2:
            self.stride = (self.stride[0], self.stride[1], 1)
        else:
            raise Exception(
                "stride can be either int or tuple of size 2. It was: {}".format(
                    self.stride.shape
                )
            )

        # padding
        if type(self.padding) == int:
            self.padding = (self.padding, self.padding, 0)
        elif len(self.padding) == 2:
            self.padding = (self.padding[0], self.padding[1], 0)
        else:
            raise Exception(
                "padding can be either int or tuple of size 2. It was: {}".format(
                    self.padding.shape
                )
            )

        # dilation
        if type(self.dilation) == int:
            self.dilation = (self.dilation, self.dilation, 1)
        elif len(self.dilation) == 2:
            self.dilation = (self.dilation[0], self.dilation[1], 1)
        else:
            raise Exception(
                "dilation can be either int or tuple of size 2. It was: {}".format(
                    self.dilation.shape
                )
            )


@dataclass
class LinearParam:
    """
    linear layer config
    """

    name: str
    type: str
    n_inputs: int
    n_outputs: int
    threshold: float
    weight_scale: float = 1.0


@dataclass
class DropoutParam:
    """
    dropout layer config
    """

    name: str
    type: str
    p: float = 0.5
    inplace: bool = False


@dataclass
class PoolParam:
    """
    pooling layer config (avg pooling using conv3d)
    """

    name: str
    type: str
    threshold: float
    kernel_size: Union[int, tuple] = 2
    padding: Union[int, tuple] = 0
    stride: Union[int, tuple] = kernel_size
    dilation: Union[int, tuple] = 1
    theta: float = 1.1

    def __post_init__(self):
        # kernel
        if type(self.kernel_size) == int:
            self.kernel_size = (self.kernel_size, self.kernel_size, 1)
        elif len(self.kernel_size) == 2:
            self.kernel_size = (self.kernel_size[0], self.kernel_size[1], 1)
        else:
            raise Exception(
                "kernelSize can only be of 1 or 2 dimension. It was: {}".format(
                    self.kernel_size.shape
                )
            )

        # stride
        if type(self.stride) == int:
            self.stride = (self.stride, self.stride, 1)
        elif len(self.stride) == 2:
            self.stride = (self.stride[0], self.stride[1], 1)
        else:
            raise Exception(
                "stride can be either int or tuple of size 2. It was: {}".format(
                    self.stride.shape
                )
            )

        # padding
        if type(self.padding) == int:
            self.padding = (self.padding, self.padding, 0)
        elif len(self.padding) == 2:
            self.padding = (self.padding[0], self.padding[1], 0)
        else:
            raise Exception(
                "padding can be either int or tuple of size 2. It was: {}".format(
                    self.padding.shape
                )
            )

        # dilation
        if type(self.dilation) == int:
            self.dilation = (self.dilation, self.dilation, 1)
        elif len(self.dilation) == 2:
            self.dilation = (self.dilation[0], self.dilation[1], 1)
        else:
            raise Exception(
                "dilation can be either int or tuple of size 2. It was: {}".format(
                    self.dilation.shape
                )
            )


class Layers(OrderedDict):
    type_dict = {
        "conv": ConvParam,
        "linear": LinearParam,
        "pooling": PoolParam,
        "dropout": DropoutParam,
    }

    def __init__(self, layer_config):
        for key, cfg in layer_config.items():
            layer_type = cfg["type"]
            try:
                self[key] = self.type_dict[layer_type](**cfg, name=key)
            except KeyError:
                raise ("layer type {} is not support".format(layer_type))

