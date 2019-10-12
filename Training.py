from sacred import Experiment
from sacred import SETTINGS
from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os

import Datasets
import Utils
import Models.UnetSpectrogramSeparator
import Models.UnetAudioSeparator
import Models.NestedUnetSpectrogramSeparator
import Models.NestedUnetAudioSeparator
import Test
import Evaluate

import functools
from tensorflow.contrib.signal import hann_window
from tensorflow.python.util import deprecation
# deprecation._print_DEPRECATION_WARNINGS = False
SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False

'''Alias Settings'''
tf.trainable_variables = tf.compat.v1.trainable_variables
tf.get_variable = tf.compat.v1.get_variable
tf.assign = tf.compat.v1.assign
tf.summary.FileWriter = tf.compat.v1.summary.FileWriter
tf.summary.scalar = tf.compat.v1.summary.scalar
tf.train.Saver = tf.compat.v1.train.Saver
tf.train.SaverDef.V2 = tf.compat.v1.train.SaverDef.V2
tf.train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer


ex = Experiment('Nested WaveUnet Training', ingredients=[config_ingredient])

@ex.config
# Executed for training, sets the seed value to the Sacred config so that Sacred fixes the Python and Numpy RNG to the same state everytime.
def set_seed():
    seed = 1337

@config_ingredient.capture
def train(model_config, experiment_id, load_model=None):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of input
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    elif model_config["network"] == "unet_spectrogram":
        separator_class = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config)
    elif model_config["network"] == "unet++":
        separator_class = Models.NestedUnetAudioSeparator.NestedUnetAudioSeparator(model_config)
    elif model_config["network"] == "unet++_spectrogram":
        separator_class = Models.NestedUnetSpectrogramSeparator.NestedUnetSpectrogramSeparator(model_config)
    else:
        raise NotImplementedError

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output
    # TODO: Add noisy_VCTK datasets for training
    # Placeholders and input normalisation
    dataset = Datasets.get_dataset(model_config, 
                                   sep_input_shape, 
                                   sep_output_shape, 
                                   partition="train")

    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

    print("Training...")

    # BUILD MODELS
    # ***Separator***
    # Sources are output in order [acc, voice] for voice separation, 
    # [bass, drums, other, vocals] for multi-instrument separation 
    separator_sources = separator_func(batch["mix"], True, 
                                       not model_config["raw_audio_loss"], 
                                       reuse=False) 

    # Supervised objective: MSE for raw audio, MAE for magnitude space (Jansson U-Net)
    separator_loss = 0
    for key in model_config["source_names"]:
        real_source = batch[key]
        sep_source = separator_sources
        # sep_source = separator_sources[key]

        if model_config["network"] == "unet_spectrogram" and not model_config["raw_audio_loss"]:
            window = functools.partial(hann_window, periodic=True)
            stfts = tf.contrib.signal.stft(tf.squeeze(real_source, 2), 
                                           frame_length=1024, 
                                           frame_step=768,
                                           fft_length=1024, 
                                           window_fn=window)
            real_mag = tf.abs(stfts)
            sub_separator_loss = 0
            if model_config["deep_supervised"]:
                for i in range(model_config["min_sub_num_layers"], model_config["num_layers"]):
                    sub_separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source[i][key]))
                sub_separator_loss /= float(model_config["num_layers"])
                separator_loss += sub_separator_loss
            else:
                separator_loss += tf.reduce_mean(tf.square(real_source - sep_source[key])) 
        else:
            sub_separator_loss = 0
            if model_config["deep_supervised"]:
                for i in range(model_config["min_sub_num_layers"], model_config["num_layers"]):
                    sub_separator_loss += tf.reduce_mean(tf.square(real_source - sep_source[i][key]))
                separator_loss += sub_separator_loss / float(model_config["num_layers"])
            else:
                separator_loss += tf.reduce_mean(tf.square(real_source - sep_source[key])) 
    # Normalise by number of sources 
    separator_loss = separator_loss / float(model_config["num_sources"]) 

    # TRAINING CONTROL VARIABLES
    global_step = tf.get_variable('global_step', [], 
                                  initializer=tf.constant_initializer(0), 
                                  trainable=False, 
                                  dtype=tf.int64)
    increment_global_step = tf.assign(global_step, global_step + 1)

    # Set up optimizers
    separator_vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
    print("Num of variables: " + str(len(tf.global_variables())))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.variable_scope("separator_solver"):
            separator_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=model_config["init_sup_sep_lr"]).minimize(separator_loss, var_list=separator_vars)

    # SUMMARIES
    tf.compat.v1.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    sup_summaries = tf.compat.v1.summary.merge_all(key='sup')

    # Start session and queue input threads
    config = tf.ConfigProto(device_count={'GPU':1})
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter(os.path.join(model_config["log_dir"], model_config["experiment_id"]), graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to continue training, if we are supposed to
    if load_model != None:
        restorer = tf.compat.v1.train.Saver(tf.global_variables(), 
                                  write_version=tf.compat.v1.train.SaverDef.V2)
        print("Num of variables: " + str(len(tf.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    saver = tf.compat.v1.train.Saver(tf.global_variables(),
                                     write_version=tf.compat.v1.train.SaverDef.V2)

    # Start training loop
    _global_step = sess.run(global_step)
    _init_step = _global_step
    for _ in range(model_config["epoch_it"]):
        # TRAIN SEPARATOR
        _, _sup_summaries = sess.run([separator_solver, sup_summaries])
        writer.add_summary(_sup_summaries, global_step=_global_step)
        if _global_step % 100 == 0:
            print('    [{}] Current step: {}'.format(model_config["network"], _global_step))

        # Increment step counter, check if maximum iterations per epoch is 
        # achieved and stop in that case
        _global_step = sess.run(increment_global_step)

    # Epoch finished - Save model
    print("Finished epoch!")
    save_path = saver.save(sess, os.path.join(model_config["model_base_dir"], model_config["experiment_id"], str(experiment_id)), global_step=int(_global_step))

    # Close session, clear computational graph
    writer.flush()
    writer.close()
    sess.close()
    tf.reset_default_graph()

    return save_path

@config_ingredient.capture
def optimise(model_config, experiment_id, model_path=None):
    epoch = 0
    best_loss = 10000
    best_model_path = None
    curr_lr = model_config["init_sup_sep_lr"]
    for i in range(3):
        worse_epochs = 0
        if i>=1:
            print("Finished first round of training, now entering fine-tuning stage")
            if i==3:
                model_config["batch_size"] *= 2
            model_config["init_sup_sep_lr"] = curr_lr
        curr_lr /= 10
        while worse_epochs < model_config["worse_epochs"]: 
            # Early stopping on validation set after a few epochs
            print("EPOCH: " + str(epoch))
            model_path = train(load_model=model_path)
            # TODO
            curr_loss = Test.test(model_config, 
                                  model_folder=os.path.join(model_config["experiment_id"],str(experiment_id)), 
                                  partition="valid", 
                                  load_model=model_path)
            epoch += 1
            if curr_loss < best_loss:
                worse_epochs = 0
                print("Performance on validation set improved from " + str(best_loss) + " to " + str(curr_loss))
                best_model_path = model_path
                best_loss = curr_loss
            else:
                worse_epochs += 1
                print("Performance on validation set worsened to " + str(curr_loss))
    print("TRAINING FINISHED - TESTING WITH BEST MODEL " + best_model_path)
    test_loss = Test.test(model_config, 
                          model_folder=os.path.join(model_config["experiment_id"], str(experiment_id)), 
                          partition="test", 
                          load_model=best_model_path)
    return best_model_path, test_loss

@ex.automain
def run(cfg):
    model_config = cfg["model_config"]
    print("SCRIPT START")
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    #model_path = "./checkpoints/unet++_10_deep_supervised/243137-88000"
    #model_path = None
    sup_model_path = "./checkpoints/unet++_12_normal-599740/599740-110000"
    # Optimize in a supervised fashion until validation loss worsens
    #sup_model_path, sup_loss = optimise(model_path=model_path)
    #print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))

    # Evaluate trained model on MUSDB
    # TODO
    Evaluate.produce_musdb_source_estimates(model_config, sup_model_path, model_config["musdb_path"], model_config["estimates_path"])
