
import numpy as np
import torch.nn.functional as F
from torch import nn

class UnetBlock(nn.Module):
    '''
    UnetBlock is composed of one downsampling and one upsampling 
    layer. UnetBlock is the basic component of WaveUnet++
    '''
    def __init__(self, channel_nums, filter_size=15, merge_filter_size=5, context=True):
        super(UnetBlock, self).__init__()
        self.in_channels = channel_nums[0]
        self.mid_channels = channel_nums[1]
        self.out_channels = channel_nums[2]
        self.padding_up = 0 if context else filter_size//2
        self.padding_down = 0 if context else merge_filter_size//2
        self.act_func = nn.LeakyReLU(0.1)
        
        # padding="valid" in tf <=> padding=0
        self.conv_up = nn.Conv1d(in_channels, mid_channels, filter_size, stride=1, padding=self.padding_up) 
        self.conv_down = nn.Conv1d(mid_channels, out_channels, merge_filter_size, stride=1, padding=self.padding_down)
    
    def forward(self, x):
        # downsampling --crop and concat--> upsampling
        #     |                                 |
        # 1D conv, 15  -------------------> 1D conv, 5
        input = x

        x = x[:,:,::2]
        x = self.conv_up(x)
        x = F.leaky_relu(x, 0.1)
        mid_out = x
        x = self.conv_down(x)
        x = F.leaky_relu(x, 0.1)
        out = F.upsample(x, scale_factor=2, mode='linear', align_corners=False)

        return out

class UnetAudioSeparator(nn.Module):
    '''
    U-Net (U-Net++) for audio seperation or speech enhancement
    Uses valid 1d convolution to predict only central part of the input,
    avoiding artifacts occur at the border
    '''
    def __init__(self, model_config):
        '''
            Init parameters for WaveUnet
        '''
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
    
    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input
        and output, so that the Unet works and has the given shape as output shape
        :return: Input_shape, output_shape, each as a list 
                 [batch_size, time_steps, channels]
        '''
        if self.context:
            # Check if desired shape is possible as output shape - go from
            # output shape towards lowest-res feature map
            rem = float(shape[1])
            
            # Output filter size
            rem = rem + self.output_filter_size - 1 #fix#

            # Upsampling blocks
            for i in range(self.num_layers):
                rem = rem + self.merge_filter_size - 1
                # out = in + in - 1 <=> in = (out+1) / 2
                rem = (rem + 1.) / 2. 
    
            # Round resulting feature map dimensions up to nearest integer
            x = np.asarray(np.ceil(rem), dtype=np.int64)
            assert(x >= 2)
    
            # Compute input and output sizes based on lowest-res feature map
            output_size = x
            input_size = x
            
            enc_sizes = list()
            dec_sizes = list()

            dec_sizes.append(output_size)
            # Extra conv
            input_size = input_size + self.filter_size - 1
            enc_sizes.append(input_size)

            # Go from centre feature map through up- and downsampling blocks
            for i in range(self.num_layers):
                output_size = 2*output_size - 1 #Upsampling
                output_size = output_size + self.merge_filter_size + 1

                input_size = 2*input_size - 1
                # input_size = input_size + self.filter_size - 1
                if i < self.num_layers - 1:
                    input_size = input_size + self.filter_size - 1
                else:
                    input_size = input_size + self.input_filter_size - 1

                enc_sizes.append(input_size)
                dec_sizes.append(output_size)

            # Output filters
            output_size = output_size - self.output_filter_size + 1
            
            input_shape = np.asarray([shape[0], input_size, self.num_channels])
            output_shape = np.asarray([shape[0], output_size, self.num_channels])
            
            return input_shape, output_shape, enc_sizes, dec_sizes
        else:
            input_shape = np.asarray([shape[0], shape[1], self.num_channels])
            output_shape = np.asarray([shape[0], shape[1], self.num_channels])
            
            return  input_shape, output_shape



    def forward(self, input, training, return_spectrogram=False, reuse=True):
        '''
        Creates symbolic computation graph of the (Nested) U-Net for a given input
        :param input: Input batch of mixtures, 3D tensor [batch_size, num_samples,                            num_channels]
        :param reuse: Whether to create new parameter variables or reuse existing
                      ones
        :return: U-Net output: List of source estimates. Each item is a 3D tensor
                               [batch_size, num_out_samples, num_channels]
        '''
        
