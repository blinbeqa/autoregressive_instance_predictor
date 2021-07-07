import torch
import torch.nn as nn


class FrozenBatchNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-5):
        super(FrozenBatchNorm, self).__init__()

        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.epsilon = epsilon

    def forward(self, x):
        scale = self.weight * (self.running_var + self.epsilon).rsqrt()
        bias = self.bias - self.running_mean * scale

        reshape_args = [1, -1] + ([1] * (x.ndimension() - 2))
        scale = scale.reshape(*reshape_args)
        bias = bias.reshape(*reshape_args)
        return x * scale + bias

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        n_conversions = 0
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            #module_output.eval()
            #print(module_output.requires_grad)
            for param in module_output.parameters():
                param.requires_grad = False
                #print(param)
            #print(module_output.requires_grad)

        for name, child in module.named_children():
            # print("name", name)
            # print("child", torch.typename(child))
            module_output.add_module(name, cls.convert_frozen_batchnorm(child))
        del module
        return module_output
