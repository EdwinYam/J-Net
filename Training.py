from sacred import Experiment
from sacred import SETTINGS
from Config import config_ingredient
from functools import partial
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import random
import time

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
deprecation._print_DEPRECATION_WARNINGS = False
SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False

#'''Alias Settings'''
#tf.trainable_variables = tf.compat.v1.trainable_variables
#tf.get_variable = tf.compat.v1.get_variable
#tf.assign = tf.compat.v1.assign
#tf.summary.FileWriter = tf.compat.v1.summary.FileWriter
#tf.summary.scalar = tf.compat.v1.summary.scalar
#tf.train.Saver = tf.compat.v1.train.Saver
#tf.train.SaverDef.V2 = tf.compat.v1.train.SaverDef.V2
#tf.train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer


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
    
    recover_sources = None if not model_config["random_recovery"] else dict()
    if model_config["random_recovery"]:
        for key in model_config["source_names"]:
            recover_sources[key] = separator_func(batch[key], True,
                                                  not model_config["raw_audio_loss"],
                                                  reuse=True)
    assert((model_config["semi_supervised"] and not model_config["context"]) or not model_config["semi_supervised"])        
    re_separator_sources = None if not model_config["semi_supervised"] else dict()
    if model_config["semi_supervised"]:
        for key in model_config["source_names"]:
            re_separator_sources[key] = separator_func(separator_sources[key], True, 
                                                       not model_config["raw_audio_loss"], 
                                                       reuse=True)
        
    # Supervised objective: MSE for raw audio, MAE for magnitude space (Jansson U-Net)
    total_loss = 0.0
    separator_loss = 0.0
    recover_loss = 0.0
    semi_loss = 0.0
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
            sub_separator_loss = 0.0
            if model_config["deep_supervised"]:
                for i in range(model_config["num_layers"]-model_config["min_sub_num_layers"]):
                    sub_separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source[i][key]))
                sub_separator_loss /= float(model_config["num_layers"]-model_config["min_sub_num_layers"])
                separator_loss += sub_separator_loss
            else:
                separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source[key])) 
                if model_config["random_recovery"]:
                    recover_loss += tf.reduce_mean(tf.abs(real_source - recover_sources[key][key]))
                if model_config["semi_supervised"]:
                    semi_loss += tf.reduce_mean(tf.abs(real_source - re_separator_sources[key][key]))
        else:
            sub_separator_loss = 0.0
            if model_config["deep_supervised"]:
                for i in range(model_config["num_layers"]-model_config["min_sub_num_layers"]):
                    sub_separator_loss += tf.reduce_mean(tf.square(real_source - sep_source[i][key]))
                separator_loss += sub_separator_loss / float(model_config["num_layers"])
            else:
                separator_loss += tf.reduce_mean(tf.square(real_source - sep_source[key]))
                if model_config["random_recovery"]:
                    recover_loss += tf.reduce_mean(tf.square(real_source - recover_sources[key][key]))
                if model_config["semi_supervised"]:
                    semi_loss += tf.reduce_mean(tf.square(real_source - re_separator_sources[key][key]))
    # Normalise by number of sources 
    separator_loss = separator_loss / float(model_config["num_sources"]) 
    recover_loss = recover_loss / float(model_config["num_sources"])
    semi_loss = semi_loss / float(model_config["num_sources"])
    if model_config["adjust_loss_ratio"]:
        semi_ratio = 0.5
        recover_ratio = 0.9
        total_loss = separator_loss + recover_ratio*recover_loss + semi_ratio*semi_loss
    else:
        total_loss = separator_loss + recover_loss + semi_loss

    # TRAINING CONTROL VARIABLES
    global_step = tf.get_variable('global_step', [], 
                                  initializer=tf.constant_initializer(0), 
                                  trainable=False, 
                                  dtype=tf.int64)
    increment_global_step = tf.assign(global_step, global_step + 1)

    # Set up optimizers
    separator_vars = Utils.getTrainableVariables("separator")
    g_vars = tf.global_variables()
    t_vars = tf.trainable_variables()
    for var in g_vars:
        if var in t_vars:
            print("    [Trainable] {}".format(var.name))
        else:
            print("    [UNTrainable] {}".format(var.name))
    #separator_vars = list()
    #for var in t_vars:
    #    if "downsample" in var.name and int(var.name.split('/')[-2].split('_')[-1]) not in model_config["random_downsample_layer"]:
    #        separator_vars.append(var)
    #        print(var.name)
    #    elif "upsample" in var.name and int(var.name.split('/')[-2].split('_')[-1]) not in model_config["random_upsample_layer"]:
    #        separator_vars.append(var)
    #        print(var.name)
    #    elif var.name.split('/')[1].split('_')[1] in model_config["source_names"]:
    #        separator_vars.append(var)
    #        print(var.name)
    #    else:
    #        print("    [Untrainable] {}".format(var.name))

    print("Num of Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
    print("Num of variables: " + str(len(tf.global_variables())))
    

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.variable_scope("separator_solver"):
            separator_solver = tf.train.AdamOptimizer(learning_rate=model_config["init_sup_sep_lr"]).minimize(total_loss, var_list=separator_vars)

    # SUMMARIES
    tf.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    tf.summary.scalar("semi_loss", semi_loss, collections=["sup"])
    tf.summary.scalar("recover_loss", recover_loss, collections=["sup"])
    sup_summaries = tf.summary.merge_all(key='sup')

    # Start session and queue input threads
    config = tf.ConfigProto(device_count={'GPU':1})
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(os.path.join(model_config["log_dir"], model_config["experiment_id"]),
                                   graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to continue training, if we are supposed to
    if load_model != None:
        restorer = tf.train.Saver(tf.global_variables(), 
                                  write_version=tf.train.SaverDef.V2)
        print("Num of variables: " + str(len(tf.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    saver = tf.train.Saver(tf.global_variables(),
                                     write_version=tf.train.SaverDef.V2)

    # Start training loop
    _global_step = sess.run(global_step)
    _init_step = _global_step
    for _ in tqdm(range(model_config["epoch_it"])):
        # TRAIN SEPARATOR
        _, _sup_summaries = sess.run([separator_solver, sup_summaries])
        writer.add_summary(_sup_summaries, global_step=_global_step)
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
    batch_size_list = [16,32,32]
    lr_list = [1e-4,1e-5,1e-6]
    for i in range(len(lr_list)):
        worse_epochs = 0
        if i>=1:
            print("Finished first round of training, now entering fine-tuning stage")
            if i==3:
                model_config["worse_epochs"] = 5
        model_config["batch_size"] = batch_size_list[i]
        model_config["init_sup_sep_lr"] = lr_list[i]
        
        while worse_epochs < model_config["worse_epochs"]: 
            # Early stopping on validation set after a few epochs
            print("EPOCH: " + str(epoch))
            model_path = train(load_model=model_path)
            # TODO
            curr_loss = Test.test(model_config, 
                                  model_folder=model_config["experiment_id"], 
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
                          model_folder=os.path.join(model_config["experiment_id"]), 
                          partition="test", 
                          load_model=best_model_path)
    return best_model_path, test_loss

@ex.automain
def run(cfg):
    model_config = cfg["model_config"]
    print("SCRIPT START")
    print("Model Configuration: {}".format(model_config))
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Optimize in a supervised fashion until validation loss worsens
    
    model_path = None
    #model_path = "checkpoints/unet-10_RNGlayer_False-107243/107243-28000"
    #model_path = "checkpoints/unet-10_RNGlayer_False-229490/229490-90000"
    #model_path = "checkpoints/unet-10_RNGlayer_True-545210/545210-14000"

    start_time = time.time()
    #sup_model_path, sup_loss = optimise(model_path=model_path)
    training_time = time.time() - start_time

    #print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))
    #print("Model Configuration: {}".format(model_config))
 
    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-220100/220100-136000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-936856/936856-160000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-782953/782953-194000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-955878/955878-132000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-912913/912913-136000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_False-610268/610268-136000" 
    #sup_model_path = "checkpoints/unet-10_RNGlayer_False-64831/64831-162000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-134822/134822-288000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-191141/191141-194000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_False-745916/745916-198000"

    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-191141/191141-194000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_False-745916/745916-198000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-930010/930010-246000"
    #sup_model_path = "checkpoints/unet-10_RNGlayer_True-191141/191141-194000"
 
    sup_model_path = "checkpoints/unet-10_RNGlayer_False-956477/956477-146000"
    model_config["semi_supervised"] = True
    # Evaluate trained model on MUSDB
    # TODO
    print(model_config["estimates_path"])
    start_time = time.time()
    Evaluate.produce_musdb_source_estimates(model_config, sup_model_path, model_config["musdb_path"], model_config["estimates_path"])
    testing_time = time.time() - start_time
    print("Model Configuration: {}".format(model_config))
    
    print('-'*20)
    print("Train time: {}".format(training_time))
    print("Test time: {}".format(testing_time))
