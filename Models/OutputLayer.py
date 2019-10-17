import tensorflow as tf

import Utils

def independent_outputs(featuremap, source_names, num_channels, filter_width, padding, activation, input_mix, residual):
    outputs = dict()
    for name in source_names:
        outputs[name] = tf.layers.conv1d(featuremap, 
                                         num_channels, 
                                         filter_width, 
                                         activation=activation, 
                                         padding=padding,
                                         name='direct_{}_conv'.format(name))
        if residual:
            outputs[name] = Utils.crop(input_mix, outputs[name].get_shape().as_list()) + outputs[name]
    
    return outputs

def difference_output(input_mix, featuremap, source_names, num_channels, filter_width, padding, activation, training):
    outputs = dict()
    sum_source = 0
    for name in source_names[:-1]:
        out = tf.layers.conv1d(featuremap, 
                               num_channels, 
                               filter_width, 
                               activation=activation, 
                               padding=padding,
                               name='direct_{}_conv'.format(name))
        outputs[name] = out
        sum_source = sum_source + out

    # Compute last source based on the others
    last_source = Utils.crop(input_mix, sum_source.get_shape().as_list()) - sum_source
    last_source = Utils.AudioClip(last_source, training)
    outputs[source_names[-1]] = last_source
    return outputs

def _independent_outputs_(featuremap, num_sources, num_channels):
    '''
    This function is the simplified version of independent_outputs, given 
    filter_width, padding, activation as default settings
    '''
    outputs = list()
    for _ in range(num_sources):
        outputs.append(tf.layers.conv1d(featuremap, 
                                        num_channels, 
                                        1, 
                                        activation=tf.tanh,
                                        padding='valid'))
    return outputs

def _difference_output_(input_mix, featuremap, num_sources, num_channels):
    outputs = list()
    last_source = input_mix
    for _ in range(num_source-1):
        out = tf.layers.conv1d(featuremap, 
                               num_channels, 
                               1,
                               activation=tf.tanh,
                               padding='valid')
        outputs.append(out)
        last_source = last_source - out
    outputs.append(last_source)
    return outputs

