                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        x

import torch
import torch.nn as nn
import torch.nn.functional as f
from convlstmcell import ConvLSTMCell, ConvLSTMCellHC, ConvTTLSTMCell, PMDCell, BlendingCell, BlendingCellNL, BlendingCellHyb
from mcnet_fkt import MotionEnc, ContentEnc, Residual, CombLayers, Discriminator, DecCnn, Encoder, Decoder
from non_local import NLBlockND
from res50_dec import Res50, Res34, Res18, DecoderResLstm
import torchvision

class NonLocalLSTM(nn.Module):
    def __init__(self,
                 # input to the model
                 input_channels,
                 # architecture of the model
                 blending_block_channels, hidden_channels, output_frames, skip_stride=None,
                 # scope of convolutional tensor-train layers
                 scope="all", scope_params={},
                 # parameters of convolutional tensor-train layers
                 cell="nllstm", cell_params={},
                 # parameters of convolutional operation
                 kernel_size=3, bias=True, res_select="res18", autoreg_select=True, number_of_instances = 1,
                 # output function and output format
                 output_sigmoid=False):
       
        super(NonLocalLSTM, self).__init__()

        ## Hyperparameters
        self.blending_block_channels = blending_block_channels
        self.hidden_channels = hidden_channels

        self.output_sigmoid = output_sigmoid
        self.autoreg_select = autoreg_select

        ## Module type of convolutional LSTM layers

        if cell == "convlstm":  # standard convolutional LSTM

            Cell = lambda in_channels, out_channels: ConvLSTMCell(
                input_channels=in_channels, hidden_channels=out_channels,
                kernel_size=kernel_size, bias=bias)

        elif cell == "convttlstm":  # convolutional tensor-train LSTM

            Cell = lambda in_channels, out_channels: ConvTTLSTMCell(
                input_channels=in_channels, hidden_channels=out_channels,
                order=cell_params["order"], steps=cell_params["steps"], ranks=cell_params["rank"],
                kernel_size=kernel_size, bias=bias)

        elif cell == "nllstm":  # Context VP LSTM

            # Cell = lambda in_channels, out_channels: PMDCell(
            #     input_channels=in_channels, input_height=64, input_width=64, hidden_channels=out_channels,
            #     kernel_size=kernel_size, bias=bias)
            Cell = lambda in_channels, out_channels: ConvLSTMCell(
                input_channels=in_channels, hidden_channels=out_channels,
                kernel_size=kernel_size, bias=bias)
            # Blending = lambda in_channels, out_channels: BlendingCellNL(
            #     input_channels=in_channels, hidden_channels=out_channels,
            #     kernel_size=kernel_size, bias=bias)

            # NonLocal = lambda in_out_channels: NLBlockND(
            #     in_channels = in_out_channels, dimension=3)



        else:
            raise NotImplementedError

        ## Construction of convolutional tensor-train LSTM network

        # stack the convolutional-LSTM layers with skip connections
        self.layers = nn.ModuleDict()
        for b in range(len(self.hidden_channels)):

                if b == 0:
                    channels = self.hidden_channels[0]
                elif b == 1:
                    channels = hidden_channels[b - 1]
                elif b == 2:
                    channels = hidden_channels[b - 1]
                elif b == 3:
                    channels = hidden_channels[b - 1]
                elif b == 4:
                    channels = hidden_channels[b - 1]
                elif b == 5:
                    channels = hidden_channels[b - 1]
                elif b == 6:
                    channels = hidden_channels[b - 1]
                elif b == 7:
                    channels = hidden_channels[b - 1]
                elif b == 8:
                    channels = hidden_channels[b - 1]
                elif b == 9:
                    channels = hidden_channels[b - 1] + self.blending_block_channels[0]
                elif b == 10:
                    channels = hidden_channels[b - 1]
                elif b == 11:
                    channels = hidden_channels[b - 1]
                elif b == 12:
                    channels = hidden_channels[b - 1]
                elif b == 13:
                    channels = hidden_channels[b - 1]
                    #channels = hidden_channels[b - 1]
                    #channels = hidden_channels[b - 1]

                lid = "b{}".format(b)  # layer
                lid_nl = "nl{}".format(0)  # layer nl

                # lid_ble = "blend{}".format(b)  # layer ID
                #
                # print("lid_ble", lid_ble)
                #print("channels", channels)

                self.layers[lid] = Cell(channels, hidden_channels[b])

                # self.layers[lid_ble] = Blending(channels, hidden_channels[b])



                # for bl in range(10):
                #     lid_ble = "b{}f{}".format(b, bl)
                #     self.layers[lid_ble] = Blending(channels, hidden_channels[b])

                #print("channels")
                #print(channels)
                #print("hidden_channels")
                #print(hidden_channels)
                #print("test output layer")
                #print(self.hidden_channels[3])
                #print(self.blending_block_channels[1])
        # self.pool_h_w = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), return_indices=True)
        # self.unpool_h_w = nn.MaxUnpool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        # self.layers["nl"] = NonLocal(hidden_channels[3])
        # self.layers["output"] = nn.Conv2d(self.hidden_channels[3] + self.blending_block_channels[1], input_channels,
        #                                   kernel_size=1, padding=0, bias=True)
        res_output_channels = 512
        if res_select == "res18":
            self.layers["res50"] = Res18(pretrained=True, number_of_instances= number_of_instances)
        elif res_select == "res34":
            self.layers["res50"] = Res34(pretrained=True)
        elif res_select == "res50":
            self.layers["res50"] = Res50(pretrained=True)
            res_output_channels = 2048

        self.layers["res-lstm"] = nn.Conv2d(res_output_channels, self.hidden_channels[0],
                                          kernel_size=1, padding=0, bias=True)
        torch.nn.init.xavier_normal_(self.layers["res-lstm"].weight)


        

        # self.layers["up"] = DecCnn(1, int((self.hidden_channels[2] + self.blending_block_channels[7])/4))

        # self.layers["output"] = nn.Conv2d(self.hidden_channels[4] + self.blending_block_channels[13], self.blending_block_channels[13],
        #                                   kernel_size=1, padding=0, bias=True)
        #print("num instaneces", number_of_instances)
        if autoreg_select:
            print("num instaneces", number_of_instances)
            self.layers["output"] = nn.Conv2d(self.hidden_channels[4] + self.blending_block_channels[13], number_of_instances,
                                            kernel_size=1, padding=0, bias=False)
            self.layers["back_to_lstm"] = nn.Conv2d(self.hidden_channels[4] + self.blending_block_channels[13], self.hidden_channels[0],
                                            kernel_size=1, padding=0, bias=False)
            torch.nn.init.xavier_normal_(self.layers["back_to_lstm"].weight)
            torch.nn.init.xavier_normal_(self.layers["output"].weight)
        else:
            self.layers["output"] = nn.Conv2d(self.hidden_channels[4] + self.blending_block_channels[13], output_frames,
                                            kernel_size=1, padding=0, bias=False)
            torch.nn.init.xavier_normal_(self.layers["output"].weight)



        self.layers["normalize"] = nn.GroupNorm(16, self.hidden_channels[0])

        layers_resnet = [512, 64]
        # self.layers["dec"] = DecoderResLstm(layers_resnet, self.blending_block_channels[13])
        
        # self.layers["output"] = nn.Conv2d(self.hidden_channels[3], input_channels,
        #                                   kernel_size=1, padding=0, bias=True)



    def forward(self, inputs, input_frames, future_frames, output_frames,
                teacher_forcing=False, scheduled_sampling_ratio=0):


        # compute the teacher forcing mask
        if teacher_forcing and scheduled_sampling_ratio > 1e-6:
            # generate the teacher_forcing mask (4-th order)
            teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                                                   torch.ones(inputs.size(0), future_frames - 1, 1, 1, 1,
                                                              device=inputs.device))
        else:  # if not teacher_forcing or scheduled_sampling_ratio < 1e-6:
            teacher_forcing = False

        actual_future_frames = future_frames

        if not self.autoreg_select:
            future_frames=0


        total_steps = input_frames + future_frames - 1
        outputs = [None] * (input_frames + future_frames)
        inputs_after_res = [None] * (input_frames + future_frames)
        inputs_after_res_targets = [None] * (future_frames)

        #print("input size", inputs.size())

        for t in range(input_frames + future_frames):

            first_step = (t == 0)

            input_ = inputs[:, t]

            # print("input before res", input_.size())
            first_dim = input_.size()[2]
            sec_dim = input_.size()[3]
            input_, res = self.layers["res50"](input_)
            #print("input after res", input_.size())
            # res = [res[7], res[2]]
            # layers_resnet = [512, 64]
            # for r in res:
            #     print("layer:", r.size())
            input_ = self.layers["res-lstm"](input_)
            input_ = self.layers["normalize"](input_)
            inputs_after_res[t] = input_
            if(t >= input_frames):
                inputs_after_res_targets[t-input_frames]=input_.detach()




            #
            #print(input_.size())
            #print("steps")
            #print(t)

        # outputs_after_lstm = [None] * (input_frames + future_frames)
        outputs_after_lstm_future_frames = [None] * future_frames

        for t in range(input_frames + future_frames):
            #print("step", t)
            first_step = (t == 0)

            if(t >= input_frames):
                #print(outputs_after_lstm_future_frames[t - input_frames].size())
                input_ = outputs_after_lstm_future_frames[t - input_frames]
            else:
                input_ = inputs_after_res[t]

            temp_a = input_

            queue = []  # previous outputs for skip connection
            for b in range(len(self.hidden_channels)):
        
                if b == 9:
                    # print("layer_forward", b)
                    # print(input_.dtype)
                    # print("shape", states_h[b].size())
                    # print("shape", states_c[b].size())
                    lid = "b{}".format(b)
                    input_ = torch.cat([input_, queue.pop(0)], dim=1)  # concat over the channels
                    input_ = self.layers[lid](input_, first_step=first_step)
                    # print("shape_out", input_.size())
                    # print("shape_out", cell_state.size())
                    # states_out_h[b] = input_
                    # states_out_c[b] = cell_state
                    queue.append(input_)
                else:
                    # print("layer_forward", b)
                    # print(input_.dtype)
                    # print("shape", states[b][0].size())
                    # print("shape", states[b][1].size())
                    lid = "b{}".format(b)  # layer ID
                    # print(lid)
                    input_ = self.layers[lid](input_, first_step=first_step)
                    # print("shape_out", input_.size())
                    # print("shape_out", cell_state.size())
                    # states_out_h[b] = input_
                    # states_out_c[b] = cell_state
                    queue.append(input_)
        
            input_ = torch.cat([input_, queue.pop(3)], dim=1)  # concat over the channels
            # map the hidden states to predictive frames (with optional sigmoid function)
            #outputs_after_lstm[t] = input_
            if self.autoreg_select:
                if(t >= input_frames - 1 and t != input_frames + future_frames - 1):
                    #print("outputs_after_lstm_future_frames ", t - (input_frames - 1))
                    outputs_after_lstm_future_frames[t - (input_frames - 1)] = self.layers["back_to_lstm"](input_)

            outputs[t] = self.layers["output"](input_)

            # outputs[t] = self.layers["dec"](outputs[t], res)


            #print("input_size", input_.size())
            #outputs[0] = self.layers["up"](input_)

            if self.output_sigmoid:
                print("sigmoid")
                outputs[t] = torch.sigmoid(outputs[t])

            outputs[t] = f.interpolate(outputs[t], size = (first_dim, sec_dim), mode='bilinear')
        
        if self.autoreg_select:
            # return the last output_frames of the outputs
            #print("autoreg")
            #print(self.autoreg_select)
            outputs = outputs[-output_frames:]

            # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
            outputs = torch.stack([outputs[t] for t in range(output_frames)], dim=1)
            res_targets = torch.stack([inputs_after_res_targets[t] for t in range(output_frames)], dim=1)
            lstm_inputs = torch.stack([outputs_after_lstm_future_frames[t] for t in range(output_frames)], dim=1)
            #print("ot", outputs.size())
            # print("res", res_targets.size())
            # print("lstm", lstm_inputs.size())
        else:
            outputs = outputs[-1:][0].unsqueeze(2)
            res_targets = inputs_after_res_targets
            lstm_inputs = outputs_after_lstm_future_frames
            #print("no autoreg")

        return outputs, res_targets, lstm_inputs
