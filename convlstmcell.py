# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils
import numpy as np


## Convolutional Tensor-Train LSTM Module
class ConvTTLSTMCell(nn.Module):

    def __init__(self,
        # interface of the Conv-TT-LSTM 
        input_channels, hidden_channels,
        # convolutional tensor-train network
        order = 3, steps = 3, ranks = 8,
        # convolutional operations
        kernel_size = 5, bias = True):
        """
        Initialization of convolutional tensor-train LSTM cell.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels)
        input_channels:  int
            Number of input channels of the input tensor.
        hidden_channels: int
            Number of hidden/output channels of the output tensor.
        Note: the number of hidden_channels is typically equal to the one of input_channels.

        (Hyper-parameters of the convolutional tensor-train format)
        order: int
            The order of convolutional tensor-train format (i.e. the number of core tensors).
            default: 3
        steps: int
            The total number of past steps used to compute the next step.
            default: 3
        ranks: int
            The ranks of convolutional tensor-train format (where all ranks are assumed to be the same).
            default: 8

        (Hyper-parameters of the convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 5
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvTTLSTMCell, self).__init__()

        ## Input/output interfaces
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        ## Convolutional tensor-train network
        self.steps = steps
        self.order = order

        self.lags = steps - order + 1

        ## Convolutional operations
        kernel_size = utils._pair(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2

        Conv2d = lambda in_channels, out_channels: nn.Conv2d(
            in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size, padding = padding, bias = bias)

        Conv3d = lambda in_channels, out_channels: nn.Conv3d(
            in_channels = in_channels, out_channels = out_channels, bias = bias,
            kernel_size = kernel_size + (self.lags, ), padding = padding + (0,))

        ## Convolutional layers
        self.layers  = nn.ModuleList()
        self.layers_ = nn.ModuleList()
        for l in range(order):
            self.layers.append(Conv2d(
                in_channels  = ranks if l < order - 1 else ranks + input_channels, 
                out_channels = ranks if l < order - 1 else 4 * hidden_channels))

            self.layers_.append(Conv3d(
                in_channels = hidden_channels, out_channels = ranks))

    def initialize(self, inputs):
        """ 
        Initialization of the hidden/cell states of the convolutional tensor-train cell.

        Arguments:
        ----------
        inputs: 4-th order tensor of size [batch_size, input_channels, height, width]
            Input tensor to the convolutional tensor-train LSTM cell.
        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states  = [torch.zeros(batch_size, self.hidden_channels, 
            height, width, device = device) for t in range(self.steps)]
        self.hidden_pointer = 0 # pointing to the position to be updated

        self.cell_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False):
        """
        Computation of the convolutional tensor-train LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            If so, both hidden and cell states are intialized to zeros tensors.
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, input_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """

        if first_step: self.initialize(inputs) # intialize the states at the first step

        ## (1) Convolutional tensor-train module
        for l in range(self.order):
            input_pointer = self.hidden_pointer if l == 0 else (input_pointer + 1) % self.steps

            input_states = self.hidden_states[input_pointer:] + self.hidden_states[:input_pointer]
            input_states = input_states[:self.lags]

            input_states = torch.stack(input_states, dim = -1)
            input_states = self.layers_[l](input_states)
            input_states = torch.squeeze(input_states, dim = -1)

            if l == 0:
                temp_states = input_states
            else: # if l > 0:
                temp_states = input_states + self.layers[l-1](temp_states)
                
        ## (2) Standard convolutional-LSTM module
        concat_conv = self.layers[-1](torch.cat([inputs, temp_states], dim = 1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states = f * self.cell_states + i * g
        outputs = o * torch.tanh(self.cell_states)
        self.hidden_states[self.hidden_pointer] = outputs
        self.hidden_pointer = (self.hidden_pointer + 1) % self.order
        
        return outputs


## Standard Convolutional-LSTM Module
class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size = 3, bias = True):
        """
        Construction of convolutional-LSTM cell.
        
        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        kernel_size = utils._pair(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2
        # print("padding")
        # print(padding)

        self.conv = nn.Conv2d(
            in_channels  = input_channels + hidden_channels, 
            out_channels = 4 * hidden_channels,
            kernel_size = kernel_size, padding = padding, bias = bias)

        torch.nn.init.xavier_normal_(self.conv.weight)

        # Note: hidden/cell states are not intialized in construction
        self.hidden_states, self.cell_state = None, None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.
        
        Arguments: 
        ----------
        inputs: a 4-th order tensor of size [batch_size, channels, height, width]
            Input tensor of convolutional-LSTM cell.`
        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states = torch.zeros(batch_size,
                                         self.hidden_channels, height, width, device=device)
        self.cell_states = torch.zeros(batch_size,
                                       self.hidden_channels, height, width, device=device)

        if inputs.type() == "torch.cuda.HalfTensor":
            self.hidden_states = self.hidden_states.half()
            self.cell_states = self.cell_states.half()


    def forward(self, inputs, first_step = False):
        """
        Computation of convolutional-LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            If so, both hidden and cell states are intialized to zeros tensors.
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """
        if first_step: self.initialize(inputs)
        #print(self.hidden_states.type())

        concat_conv = self.conv(torch.cat([inputs, self.hidden_states], dim = 1))
        #concat_conv2 = self.conv(torch.cat([inputs, self.hidden_states], dim = 2))
        #print("shape of concat_conv")
        #print(concat_conv.shape)
        #print("shape of concat_conv2")
        #print(concat_conv2.shape)

        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states   = f * self.cell_states + i * g
        self.hidden_states = o * torch.tanh(self.cell_states)
        # self.cell_state.detach()
        # self.hidden_states.detach

        
        return self.hidden_states


class ConvLSTMCellHC(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size=3, bias=True):
        """
        Construction of convolutional-LSTM cell.

        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvLSTMCellHC, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        kernel_size = utils._pair(kernel_size)
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        # print("padding")
        # print(padding)

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)

        torch.nn.init.xavier_normal_(self.conv.weight)

        # Note: hidden/cell states are not intialized in construction
        self.hidden_states, self.cell_state = None, None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, channels, height, width]
            Input tensor of convolutional-LSTM cell.`
        """
        device = inputs.device  # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states = torch.zeros(batch_size,
                                         self.hidden_channels, height, width, device=device)
        self.cell_states = torch.zeros(batch_size,
                                       self.hidden_channels, height, width, device=device)

    def forward(self, inputs, first_step=False):
        """
        Computation of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width]
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence.
            If so, both hidden and cell states are intialized to zeros tensors.

        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """
        if first_step:
            self.initialize(inputs)
        else:
            self.hidden_states = self.hidden_states.detach().clone()
            self.cell_states = self.cell_states.detach().clone()

        concat_conv = self.conv(torch.cat([inputs, self.hidden_states], dim=1))
        # concat_conv2 = self.conv(torch.cat([inputs, self.hidden_states], dim = 2))
        # print("shape of concat_conv")
        # print(concat_conv.shape)
        # print("shape of concat_conv2")
        # print(concat_conv2.shape)

        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states = f * self.cell_states + i * g
        self.hidden_states = o * torch.tanh(self.cell_states)

        return self.hidden_states


## Blending Block Module
class BlendingCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size=3, bias=True):
        """
        Construction of PMD cell.

        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.
        batch_s: int
            Batch size if input sequence

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(BlendingCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        #print("parametrat")



        kernel_size = utils._pair(kernel_size)
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        #print(kernel_size)
        #print(padding)


        self.conv_sum = nn.Conv2d(
            in_channels=3 * hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1, padding=0, bias=bias)


        self.hidden_states_sum = None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, channels, height, width]
            Input tensor of convolutional-LSTM cell.
        """
        device = inputs.device  # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros


        self.hidden_states_sum = torch.zeros(batch_size,
                                             self.hidden_channels, height, width, device=device)

    def forward(self, inputs, first_step=True):
        """
        Computation of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width]
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence.
            If so, both hidden and cell states are intialized to zeros tensors.

        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """
        if first_step: self.initialize(inputs)



        self.hidden_states_sum = self.conv_sum(inputs)

        return self.hidden_states_sum

class BlendingCellNL(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size=3, bias=True):
        """
        Construction of PMD cell.

        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.
        batch_s: int
            Batch size if input sequence

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(BlendingCellNL, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        #print("parametrat")



        kernel_size = utils._pair(kernel_size)
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        #print(kernel_size)
        #print(padding)


        self.conv_sum = nn.Conv2d(
            in_channels= hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1, padding=0, bias=bias)


        self.hidden_states_sum = None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, channels, height, width]
            Input tensor of convolutional-LSTM cell.
        """
        device = inputs.device  # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros


        self.hidden_states_sum = torch.zeros(batch_size,
                                             self.hidden_channels, height, width, device=device)

    def forward(self, inputs, first_step=True):
        """
        Computation of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width]
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence.
            If so, both hidden and cell states are intialized to zeros tensors.

        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """
        if first_step: self.initialize(inputs)



        self.hidden_states_sum = self.conv_sum(inputs)

        return self.hidden_states_sum

## Blending Block Module
class BlendingCellHyb(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size=3, bias=True):
        """
        Construction of PMD cell.

        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.
        batch_s: int
            Batch size if input sequence

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(BlendingCellHyb, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        #print("parametrat")



        kernel_size = utils._pair(kernel_size)
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        #print(kernel_size)
        #print(padding)


        self.conv_sum = nn.Conv2d(
            in_channels=2 * hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1, padding=0, bias=bias)


        self.hidden_states_sum = None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, channels, height, width]
            Input tensor of convolutional-LSTM cell.
        """
        device = inputs.device  # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros


        self.hidden_states_sum = torch.zeros(batch_size,
                                             self.hidden_channels, height, width, device=device)

    def forward(self, inputs, first_step=True):
        """
        Computation of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width]
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence.
            If so, both hidden and cell states are intialized to zeros tensors.

        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """
        if first_step: self.initialize(inputs)



        self.hidden_states_sum = self.conv_sum(inputs)

        return self.hidden_states_sum

## Parallel Multi-Dimensional Module
class PMDCell(nn.Module):

    def __init__(self, input_channels, input_height, input_width, hidden_channels, kernel_size=3, bias=True):
        """
        Construction of PMD cell.

        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.
        batch_s: int
            Batch size if input sequence

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(PMDCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        #print("parametrat")



        kernel_size = utils._pair(kernel_size)
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        #print(kernel_size)
        #print(padding)

        self.conv_t_min = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_h_min = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        # self.conv_h_plus = nn.Conv2d(
        #     in_channels=input_channels + hidden_channels,
        #     out_channels=4 * hidden_channels,
        #     kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_w_min = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        # self.conv_w_plus = nn.Conv2d(
        #     in_channels=input_channels + hidden_channels,
        #     out_channels=4 * hidden_channels,
        #     kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_sum = nn.Conv2d(
            in_channels=3 * hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1, padding=0, bias=bias)

        # Note: hidden/cell states are not intialized in construction
        self.hidden_states_t_min, self.cell_states_t_min = None, None
        self.hidden_states_h_min, self.cell_states_h_min = None, None
        self.hidden_states_w_min, self.cell_states_w_min = None, None
        # self.hidden_states_h_plus, self.cell_states_h_plus = None, None
        # self.hidden_states_w_plus, self.cell_states_w_plus = None, None
        self.hidden_states_sum = None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, channels, height, width]
            Input tensor of convolutional-LSTM cell.
        """
        device = inputs.device  # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros

        self.hidden_states_t_min = torch.zeros(batch_size,
                                               self.hidden_channels, height, width, device=device)
        self.cell_states_t_min = torch.zeros(batch_size,
                                             self.hidden_channels, height, width, device=device)

        self.hidden_states_h_min = torch.zeros(height,
                                               self.hidden_channels, batch_size, width, device=device)
        self.cell_states_h_min = torch.zeros(height,
                                             self.hidden_channels, batch_size, width, device=device)
        # self.hidden_states_h_plus = torch.zeros(height,
        #                                         self.hidden_channels, batch_size, width, device=device)
        # self.cell_states_h_plus = torch.zeros(height,
        #                                       self.hidden_channels, batch_size, width, device=device)
        self.hidden_states_w_min = torch.zeros(width,
                                               self.hidden_channels, height, batch_size, device=device)
        self.cell_states_w_min = torch.zeros(width,
                                             self.hidden_channels, height, batch_size, device=device)
        # self.hidden_states_w_plus = torch.zeros(width,
        #                                         self.hidden_channels, height, batch_size, device=device)
        # self.cell_states_w_plus = torch.zeros(width,
        #                                         self.hidden_channels, height, batch_size, device=device)

        self.hidden_states_sum = torch.zeros(batch_size,
                                             self.hidden_channels, height, width, device=device)

    def forward(self, inputs, first_step=False):
        """
        Computation of convolutional-LSTM cell.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width]
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence.
            If so, both hidden and cell states are intialized to zeros tensors.

        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """
        if first_step: self.initialize(inputs)


        concat_conv_t_min = self.conv_t_min(torch.cat([inputs, self.hidden_states_t_min], dim=1))
        # concat_conv2 = self.conv(torch.cat([inputs, self.hidden_states], dim = 2))
        # print("shape of concat_conv")
        # print(concat_conv.shape)
        # print("shape of concat_conv2")
        # print(concat_conv2.shape)
        #print("blinitest")
        #print(inputs.shape)
        cc_i_t_min, cc_f_t_min, cc_o_t_min, cc_g_t_min = torch.split(concat_conv_t_min, self.hidden_channels, dim=1)

        i_t_min = torch.sigmoid(cc_i_t_min)
        f_t_min = torch.sigmoid(cc_f_t_min)
        o_t_min = torch.sigmoid(cc_o_t_min)
        g_t_min = torch.tanh(cc_g_t_min)

        self.cell_states_t_min = f_t_min * self.cell_states_t_min + i_t_min * g_t_min
        self.hidden_states_t_min = o_t_min * torch.tanh(self.cell_states_t_min)

        #h_min
        concat_conv_h_min = self.conv_h_min(torch.cat([inputs.permute(2, 1, 0, 3), self.hidden_states_h_min], dim=1))

        cc_i_h_min, cc_f_h_min, cc_o_h_min, cc_g_h_min = torch.split(concat_conv_h_min, self.hidden_channels,
                                                                     dim=1)

        i_h_min = torch.sigmoid(cc_i_h_min)
        f_h_min = torch.sigmoid(cc_f_h_min)
        o_h_min = torch.sigmoid(cc_o_h_min)
        g_h_min = torch.tanh(cc_g_h_min)

        self.cell_states_h_min = f_h_min * self.cell_states_h_min + i_h_min * g_h_min
        self.hidden_states_h_min = o_h_min * torch.tanh(self.cell_states_h_min)

        #h_plus with DWS
        concat_conv_h_min = self.conv_h_min(torch.cat([inputs.permute(2, 1, 0, 3).flip(0), self.hidden_states_h_min], dim=1))

        cc_i_h_min, cc_f_h_min, cc_o_h_min, cc_g_h_min = torch.split(concat_conv_h_min, self.hidden_channels,
                                                                     dim=1)

        i_h_min = torch.sigmoid(cc_i_h_min)
        f_h_min = torch.sigmoid(cc_f_h_min)
        o_h_min = torch.sigmoid(cc_o_h_min)
        g_h_min = torch.tanh(cc_g_h_min)

        self.cell_states_h_min = f_h_min * self.cell_states_h_min + i_h_min * g_h_min
        self.hidden_states_h_min = o_h_min * torch.tanh(self.cell_states_h_min)


        # # h_plus
        # concat_conv_h_plus = self.conv_h_plus(torch.cat([inputs.permute(2, 1, 0, 3).flip(0), self.hidden_states_h_min], dim=1))
        #
        # cc_i_h_plus, cc_f_h_plus, cc_o_h_plus, cc_g_h_plus = torch.split(concat_conv_h_plus, self.hidden_channels,
        #                                                              dim=1)
        #
        # i_h_plus = torch.sigmoid(cc_i_h_plus)
        # f_h_plus = torch.sigmoid(cc_f_h_plus)
        # o_h_plus = torch.sigmoid(cc_o_h_plus)
        # g_h_plus = torch.tanh(cc_g_h_plus)
        #
        # self.cell_states_h_plus = f_h_plus * self.cell_states_h_plus + i_h_plus * g_h_plus
        # self.hidden_states_h_plus = o_h_plus * torch.tanh(self.cell_states_h_plus)

        #print("hidden_states_t_min")
        #print(self.hidden_states_t_min.shape)
        #print("hidden_states_h_plus")
        #print(self.hidden_states_h_plus.permute(2, 1, 0, 3).shape)

        #w_min
        concat_conv_w_min = self.conv_w_min(torch.cat([inputs.permute(3, 1, 2, 0), self.hidden_states_w_min], dim=1))

        cc_i_w_min, cc_f_w_min, cc_o_w_min, cc_g_w_min = torch.split(concat_conv_w_min, self.hidden_channels,
                                                                     dim=1)

        i_w_min = torch.sigmoid(cc_i_w_min)
        f_w_min = torch.sigmoid(cc_f_w_min)
        o_w_min = torch.sigmoid(cc_o_w_min)
        g_w_min = torch.tanh(cc_g_w_min)

        self.cell_states_w_min = f_w_min * self.cell_states_w_min + i_w_min * g_w_min
        self.hidden_states_w_min = o_w_min * torch.tanh(self.cell_states_w_min)

        # w_plus with DWS
        concat_conv_w_min = self.conv_w_min(torch.cat([inputs.permute(3, 1, 2, 0).flip(0), self.hidden_states_w_min], dim=1))

        cc_i_w_min, cc_f_w_min, cc_o_w_min, cc_g_w_min = torch.split(concat_conv_w_min, self.hidden_channels,
                                                                     dim=1)

        i_w_min = torch.sigmoid(cc_i_w_min)
        f_w_min = torch.sigmoid(cc_f_w_min)
        o_w_min = torch.sigmoid(cc_o_w_min)
        g_w_min = torch.tanh(cc_g_w_min)

        self.cell_states_w_min = f_w_min * self.cell_states_w_min + i_w_min * g_w_min
        self.hidden_states_w_min = o_w_min * torch.tanh(self.cell_states_w_min)

        # w_plus
        # concat_conv_w_plus = self.conv_w_plus(torch.cat([inputs.permute(3, 1, 2, 0).flip(0), self.hidden_states_w_min], dim=1))
        #
        # cc_i_w_plus, cc_f_w_plus, cc_o_w_plus, cc_g_w_plus = torch.split(concat_conv_w_plus, self.hidden_channels,
        #                                                                  dim=1)
        #
        # i_w_plus = torch.sigmoid(cc_i_w_plus)
        # f_w_plus = torch.sigmoid(cc_f_w_plus)
        # o_w_plus = torch.sigmoid(cc_o_w_plus)
        # g_w_plus = torch.tanh(cc_g_w_plus)
        #
        # self.cell_states_w_plus = f_w_plus * self.cell_states_w_plus + i_w_plus * g_w_plus
        # self.hidden_states_w_plus = o_w_plus * torch.tanh(self.cell_states_w_plus)

        concat_all = torch.cat([self.hidden_states_t_min, self.hidden_states_h_min.permute(2, 1, 0, 3)], dim=1)
        #concat_all = torch.cat([concat_all, self.hidden_states_h_plus.permute(2, 1, 0, 3)], dim=1)
        concat_all = torch.cat([concat_all, self.hidden_states_w_min.permute(3, 1, 2, 0)], dim=1)
        #concat_all = torch.cat([concat_all, self.hidden_states_w_plus.permute(3, 1, 2, 0)], dim=1)

        self.hidden_states_sum = self.conv_sum(concat_all)

        return self.hidden_states_sum