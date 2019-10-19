import tensorflow as tf

import Models.InterpolationLayer
import Utils
from Utils import LeakyReLU
import numpy as np
import Models.OutputLayer

'''
Alias settings:
    Since tensorflow keep changing the relative path/location of some certain 
    functions, thus, we simply set some aliases to avoid annoying tensorflow 
    deprecation warning
'''
#tf.variable_scope = tf.compat.v1.variable_scope
#tf.image.resize_bilinear = tf.compat.v1.image.resize_bilinear
#tf.variable_scope = tf.compat.v1.variable_scope

class UnetAudioSeparator:
    '''
    U-Net separator network for singing voice separation.
    Uses valid convolutions, so it predicts for the centre part of the input - only certain input and output shapes are therefore possible (see getpadding function)
    '''

    def __init__(self, model_config):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network 
        '''
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.num_increase_filters = model_config["num_increase_filters"]
        self.filter_size = model_config["filter_size"]
        self.merge_filter_size = model_config["merge_filter_size"]
        self.input_filter_size = model_config["input_filter_size"]
        self.output_filter_size = model_config["output_filter_size"]
        self.upsampling = model_config["upsampling"]
        self.output_type = model_config["output_type"]
        self.context = model_config["context"]
        self.padding = "valid" if model_config["context"] else "same"
        self.source_names = model_config["source_names"]
        self.num_channels = 1 if model_config["mono_downmix"] else 2
        self.output_activation = model_config["output_activation"]
        self.add_random_layer = model_config["add_random_layer"]
        self.min_skip_num_layers = model_config["min_skip_num_layers"]
        self.max_skip_num_layers = model_config["max_skip_num_layers"]
        self.random_downsample_layer = [ (i in model_config["random_downsample_layer"]) for i in range(self.num_layers) ]
        self.random_upsample_layer = [ (i in model_config["random_upsample_layer"]) for i in range(self.num_layers) ]
        self.remove_random = model_config["remove_random"]
        self.use_meanvar = model_config["use_meanvar"]
        self.add_multires_block = model_config["add_multires_block"]
        self.add_res_path = model_config["add_res_path"]
        self.skip_layer = model_config["skip_layer"]
        # self.residual = model_config["residual"]
        self.residual = False

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape 
        :return: Input_shape, output_shape, where each is a list [batch_size, time_steps, channels]
        '''

        if self.context:
            # Check if desired shape is possible as output shape - go from output shape towards lowest-res feature map
            rem = float(shape[1]) # Cut off batch size number and channel

            # Output filter size
            rem = rem - self.output_filter_size + 1

            # Upsampling blocks
            for i in range(self.num_layers):
                rem = rem + self.merge_filter_size - 1
                rem = (rem + 1.) / 2.# out = in + in - 1 <=> in = (out+1)/

            # Round resulting feature map dimensions up to nearest integer
            x = np.asarray(np.ceil(rem),dtype=np.int64)
            assert(x >= 2)

            # Compute input and output shapes based on lowest-res feature map
            output_shape = x
            input_shape = x

            # Extra conv
            input_shape = input_shape + self.filter_size - 1

            # Go from centre feature map through up- and downsampling blocks
            for i in range(self.num_layers):
                output_shape = 2*output_shape - 1 #Upsampling
                if not (self.remove_random and self.random_upsample_layer[i]):
                    output_shape = output_shape - self.merge_filter_size + 1 # Conv

                input_shape = 2*input_shape - 1 # Decimation
                if not (self.remove_random and self.random_downsample_layer[self.num_layers-i-1]):
                    if i < self.num_layers - 1:
                        input_shape = input_shape + self.filter_size - 1 # Conv
                    else:
                        input_shape = input_shape + self.input_filter_size - 1

            # Output filters
            output_shape = output_shape - self.output_filter_size + 1

            input_shape = np.concatenate([[shape[0]], [input_shape], [self.num_channels]])
            output_shape = np.concatenate([[shape[0]], [output_shape], [self.num_channels]])

            return input_shape, output_shape
        else:
            return [shape[0], shape[1], self.num_channels], [shape[0], shape[1], self.num_channels]

    def get_output(self, input, training, return_spectrogram=False, reuse=True, use_discriminator=False):
        '''
        Creates symbolic computation graph of the U-Net for a given input batch
        :param input: Input batch of mixtures, 3D tensor [batch_size, num_samples, num_channels]
        :param reuse: Whether to create new parameter variables or reuse existing ones
        :return: U-Net output: List of source estimates. Each item is a 3D tensor [batch_size, num_out_samples, num_channels]
        '''
        with tf.variable_scope("separator", reuse=reuse):
            enc_outputs = list()
            current_layer = input

            # Down-convolution: Repeat strided conv
            for i in range(self.num_layers):
                if self.random_downsample_layer[i] and self.remove_random:
                    assert(self.add_random_layer)
                else:
                    current_layer = tf.layers.conv1d(current_layer, 
                                                     self.num_initial_filters + (self.num_increase_filters * i), 
                                                     self.filter_size, 
                                                     strides=1, 
                                                     activation=LeakyReLU, 
                                                     padding=self.padding, 
                                                     name='downsample_conv_{}'.format(i), 
                                                     trainable= not(self.add_random_layer and self.random_downsample_layer[i])) # out = in - filter + 1
                    if self.add_multires_block:
                        curr_layer = current_layer
                        _current_layer = tf.layers.conv1d(curr_layer, 
                                                         2*(self.num_initial_filters + (self.num_increase_filters * i)), 
                                                         1, 
                                                         strides=1, 
                                                         activation=LeakyReLU, 
                                                         padding='same', 
                                                         name='downsample_multires_conv_{}'.format(i), 
                                                         trainable= not(self.add_random_layer and self.random_downsample_layer[i])) # out = in - filter + 1
                        for j in range(2-1):
                            curr_layer = tf.layers.conv1d(curr_layer, 
                                                              self.num_initial_filters + (self.num_increase_filters * i), 
                                                              self.filter_size, 
                                                              strides=1, 
                                                              activation=LeakyReLU, 
                                                              padding='same', 
                                                              name='downsample_multires_{}_conv_{}'.format(j,i), 
                                                              trainable= not(self.add_random_layer and self.random_downsample_layer[i])) # out = in - filter + 1            
                            current_layer = Utils.crop_and_concat(current_layer, curr_layer, match_feature_dim=False)
                        current_layer = current_layer + _current_layer


                if self.add_res_path and (i in self.skip_layer):
                    res_path = current_layer
                    for j in range(3):
                        res_path = tf.layers.conv1d(res_path,
                                                    self.num_initial_filters + (self.num_increase_filters * i),
                                                    self.filter_size,
                                                    strides=1,
                                                    activation=LeakyReLU,
                                                    padding='same',
                                                    name='downsample_res_{}_conv_{}'.format(j,i))
                        sub_res_path = tf.layers.conv1d(res_path,
                                                        self.num_initial_filters + (self.num_increase_filters * i),
                                                        1,
                                                        strides=1,
                                                        activation=LeakyReLU,
                                                        padding='same',
                                                        name='downsample_sub_res_{}_conv_{}'.format(j,i))
                        res_path = res_path + sub_res_path
                    enc_outputs.append(res_path)
                else:
                    enc_outputs.append(current_layer)
                print("    [unet] downconv{} shape: {}".format(i+1, enc_outputs[i].get_shape().as_list()))
                current_layer = current_layer[:,::2,:] # Decimate by factor of 2 # out = (in-1)/2 + 1

            current_layer = tf.layers.conv1d(current_layer, 
                                             self.num_initial_filters + (self.num_increase_filters * self.num_layers),
                                             self.filter_size,
                                             activation=LeakyReLU,
                                             padding=self.padding, 
                                             name='downsample_conv_{}'.format(self.num_layers)) # One more conv here since we need to compute features after last decimation


            # Feature map here shall be X along one dimension

            # Upconvolution
            for i in range(self.num_layers):
                #UPSAMPLING
                current_layer = tf.expand_dims(current_layer, axis=1)
                if self.upsampling == 'learned':
                    # Learned interpolation between two neighbouring time positions by using a convolution filter of width 2, and inserting the responses in the middle of the two respective inputs
                    current_layer = Models.InterpolationLayer.learned_interpolation_layer(current_layer, self.padding, i)
                else:
                    if self.context:
                        current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2] * 2 - 1], align_corners=True)
                    else:
                        current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2]*2]) # out = in + in - 1
                current_layer = tf.squeeze(current_layer, axis=1)
                # UPSAMPLING FINISHED

                # assert(enc_outputs[-i-1].get_shape().as_list()[1] == current_layer.get_shape().as_list()[1] or self.context) #No cropping should be necessary unless we are using context
                if self.num_layers-i-1 in self.skip_layer:
                    current_layer = Utils.crop_and_concat(enc_outputs[-i-1], current_layer, match_feature_dim=False)
                elif (self.num_layers-i < self.max_skip_num_layers and self.num_layers-i > self.min_skip_num_layers) and self.add_random_layer:
                    current_layer = Utils.crop_and_concat(enc_outputs[-i-1], current_layer, match_feature_dim=False)
                
                print("    [unet] upconv_{} shape: {}".format(self.num_layers-i, current_layer.get_shape().as_list()))
                if self.random_upsample_layer[self.num_layers-i-1] and self.remove_random:
                    continue

                current_layer = tf.layers.conv1d(current_layer, 
                                                 self.num_initial_filters + (self.num_increase_filters * (self.num_layers - i - 1)), 
                                                 self.merge_filter_size,
                                                 name="upsample_conv_{}".format(self.num_layers-i-1),
                                                 activation=LeakyReLU,
                                                 padding=self.padding,
                                                 trainable=not(self.add_random_layer and self.random_upsample_layer[self.num_layers-i-1]))  # out = in - filter + 1
                if self.add_multires_block:
                    curr_layer = current_layer
                    _current_layer = tf.layers.conv1d(curr_layer, 
                                                      2*(self.num_initial_filters + (self.num_increase_filters * (self.num_layers - i - 1))), 
                                                      1, 
                                                      strides=1, 
                                                      activation=LeakyReLU, 
                                                      padding='same', 
                                                      name='upsample_multires_conv_{}'.format(i)) # out = in - filter + 1
                    for j in range(2-1):
                        curr_layer = tf.layers.conv1d(curr_layer, 
                                                      self.num_initial_filters + (self.num_increase_filters * (self.num_layers - i - 1)), 
                                                      self.merge_filter_size, 
                                                      strides=1, 
                                                      activation=LeakyReLU, 
                                                      padding='same', 
                                                      name='upsample_multires_{}_conv_{}'.format(j,i)) # out = in - filter + 1            
                        current_layer = Utils.crop_and_concat(current_layer, curr_layer, match_feature_dim=False)
                    current_layer = current_layer + _current_layer

            current_layer = Utils.crop_and_concat(input, current_layer, match_feature_dim=False)

            # Output layer
            # Determine output activation function
            if self.output_activation == "tanh":
                out_activation = tf.tanh
            elif self.output_activation == "linear":
                out_activation = lambda x: Utils.AudioClip(x, training)
            else:
                raise NotImplementedError

            if self.output_type == "direct":
                return Models.OutputLayer.independent_outputs(current_layer, self.source_names, self.num_channels, self.output_filter_size, self.padding, out_activation, input, residual=self.residual)
            elif self.output_type == "difference":
                assert(not self.residual)
                cropped_input = Utils.crop(input,current_layer.get_shape().as_list(), match_feature_dim=False)
                return Models.OutputLayer.difference_output(cropped_input, current_layer, self.source_names, self.num_channels, self.output_filter_size, self.padding, out_activation, training)
            else:
                raise NotImplementedError
