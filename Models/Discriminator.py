import tensorflow as tf

from Utils import LeakyReLU

class Nested_Discriminator:
    '''
    Share parameter with the downsampling part of U-net separator
    '''
    def __init__(self, model_config, source_name):
        '''
        Initialize Discriminator
        '''
        self.source_name = source_name
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.num_increase_filters = int(model_config["num_increase_filters"] / 2)
        self.filter_size = model_config["filter_size"]
        self.padding = "valid" if model_config["context"] else "same"
        self.d_min_sub_num_layers = model_config["d_min_sub_num_layers"]

        
    def get_output(self, input, training, reuse=True):
        with tf.variable_scope("{}_discriminator".format(self.source_name), reuse=reuse):
            enc_outputs = list()
            for i in range(self.d_min_sub_num_layers, self.num_layers):
                current_layer = input[i]
                for j in range(i, self.num_layers):
                    current_layer = current_layer[:,::2,:]
                    current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_increase_filters * j), self.filter_size, strides=1, activation=LeakyReLU, padding=self.padding, name='d_downconv_{}_{}'.format(i,j))
                enc_outputs.append(current_layer)
                print("    [disc] d_downconv_{} shape: {}".format(i+1, enc_outputs[i-self.d_min_sub_num_layers].get_shape().as_list()))
            
            # current_layer = input[self.num_layers]
            # current_layer = tf.layers.conv1d(current_layer, 1, self.filter_size, strides=1, activation=LeakyReLU, padding=self.padding, name='d_downconv_{}_{}'.format(i,self.num_layers-i))
            enc_outputs.append(input[self.num_layers])

            current_layer = tf.layers.conv1d(tf.concat(enc_outputs, axis=2), 1, self.filter_size, strides=2, activation=LeakyReLU, padding=self.padding, name='d_merge_conv')
            current_layer = tf.contrib.layers.flatten(current_layer)
            print("    [disc] Final dim of embedding before fc: {}".format(current_layer.get_shape().as_list()))
            hidden = tf.layers.dense(current_layer, 32, activation=LeakyReLU, use_bias=True, name='d_fc_0')
            logits = tf.layers.dense(hidden, 1, activation=None, use_bias=False, name='d_fc_1')
        
            return logits
            



