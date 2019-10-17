from sacred import Experiment
from sacred import SETTINGS
from Config import config_ingredient
from functools import partial
import tensorflow as tf
import numpy as np
import os
import random

import Datasets
import Utils
import Models.UnetSpectrogramSeparator
import Models.UnetAudioSeparator
import Models.NestedUnetSpectrogramSeparator
import Models.NestedUnetAudioSeparator
import Models.Discriminator
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
    # Disciminator shares parameter with the downsampling/encoder part of
    # the separator, in order to utilize the embedding learned from downsampling
    discriminator_func = partial(separator_func, use_discriminator=True)
   
    tail_discriminators, tail_discriminators_func = dict(), dict()
    if model_config["discriminated"]:
        assert(not model_config["context"])
        tail_discriminators = { name: Models.Discriminator.Nested_Discriminator(model_config, name) for name in model_config["source_names"] }
        tail_discriminators_func = { name: tail_discriminators[name].get_output for name in model_config["source_names"] }

    # TODO: Add noisy_VCTK datasets for training
    # Placeholders and input normalisation
    dataset = Datasets.get_dataset(model_config, 
                                   sep_input_shape, 
                                   sep_output_shape, 
                                   partition="train")

    iterator = dataset.make_one_shot_iterator()
    _batch = iterator.get_next()
    batch = dict()
    recover_source = model_config["random_recovery"] and random.uniform(0,1) > 0.3
    if recover_source:
        chosen_source = model_config["source_names"][random.randint(0,model_config["num_sources"]-1)]
        batch["mix"] = _batch[chosen_source]
        for name in model_config["source_names"]:
            if name != chosen_source:
                batch[name] = tf.zeros_like(_batch[name])
            else:
                batch[name] = _batch[name]
    else:
        batch = _batch

    print("Training...")

    # BUILD MODELS
    # ***Separator***
    # Sources are output in order [acc, voice] for voice separation, 
    # [bass, drums, other, vocals] for multi-instrument separation 
    separator_sources = separator_func(batch["mix"], True, 
                                       not model_config["raw_audio_loss"], 
                                       reuse=False)
    
    assert((model_config["semi_supervised"] and not model_config["context"]) or not model_config["semi_supervised"])

    # Supervised objective: MSE for raw audio, MAE for magnitude space (Jansson U-Net)
    separator_loss = 0.0
    d_loss = 0.0
    d_rl_loss = 0.0
    d_fk_loss = 0.0
    g_adv_loss = 0.0
    for index, key in enumerate(model_config["source_names"]):
        real_source = batch[key]
        sep_source = separator_sources
        re_sep_source = separator_func(separator_sources[key], True, not model_config["raw_audio_loss"], reuse=True) if model_config["semi_supervised"] else None
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
                for i in range(model_config["num_layers"]-model_config["min_sub_num_layers"]):
                    sub_separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source[i][key]))
                    if model_config["semi_supervised"]:
                        sub_separator_loss += model_config["semi_loss_ratio"] * tf.reduce_mean(tf.abs(real_mag-re_sep_source[i][key]))
                sub_separator_loss /= float(model_config["num_layers"])
                separator_loss += sub_separator_loss
            else:
                separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source[key])) 
                if model_config["semi_supervised"]:
                    separator_loss += model_config["semi_loss_ratio"] * tf.reduce_mean(tf.abs(real_mag-re_sep_source[key]))
        else:
            sub_separator_loss = 0.0
            if model_config["deep_supervised"]:
                for i in range(model_config["num_layers"]-model_config["min_sub_num_layers"]):
                    sub_separator_loss += tf.reduce_mean(tf.square(real_source - sep_source[i][key]))
                    if model_config["semi_supervised"]:
                        sub_separator_loss += tf.reduce_mean(tf.square(real_source - re_sep_source[i][key]))
                separator_loss += sub_separator_loss / float(model_config["num_layers"])
            else:
                separator_loss += tf.reduce_mean(tf.square(real_source - sep_source[key])) 
                if model_config["semi_supervised"]:
                    separator_loss += tf.reduce_mean(tf.square(real_source - re_sep_source[key]))
            if model_config["discriminated"]:
                d_rl_embeddings = discriminator_func(batch[key], True, not model_config["raw_audio_loss"], reuse=True)
                d_rl_logits = tail_discriminators_func[key](d_rl_embeddings, True, reuse=False)
                d_rl_loss += tf.reduce_mean(tf.squared_difference(d_rl_logits, 1.))
                
                d_fk_embeddings = discriminator_func(sep_source[key], True, not model_config["raw_audio_loss"], reuse=True)
                d_fk_logits = tail_discriminators_func[key](d_fk_embeddings, True, reuse=True)
                d_fk_loss += tf.reduce_mean(tf.squared_difference(d_fk_logits, 0.))

                g_adv_loss += tf.reduce_mean(tf.squared_difference(d_fk_logits, 1.))

    # Normalise by number of sources 
    separator_loss = separator_loss / float(model_config["num_sources"]) 
    if model_config["discriminated"]:
        g_adv_loss = model_config["g_loss_weight"] * g_adv_loss / float(model_config["num_sources"])
        d_loss = (d_rl_loss + d_fk_loss) / float(model_config["num_sources"])

    # TRAINING CONTROL VARIABLES
    global_step = tf.get_variable('global_step', [], 
                                  initializer=tf.constant_initializer(0), 
                                  trainable=False, 
                                  dtype=tf.int64)
    increment_global_step = tf.assign(global_step, global_step + 1)
    
    g_global_step = tf.get_variable('g_global_step', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False,
                                    dtype=tf.int64)
    increment_g_global_step = tf.assign(g_global_step, g_global_step + 1)

    d_global_step = tf.get_variable('d_global_step', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False,
                                    dtype=tf.int64)
    increment_d_global_step = tf.assign(d_global_step, d_global_step + 1)

    # Set up optimizers
    separator_vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
    print("Num of variables: " + str(len(tf.global_variables())))
   
    d_vars = list()
    g_vars = list()
    if model_config["discriminated"]:
        t_vars = tf.trainable_variables()
        for var in t_vars:
            if "discriminator" in var.name:
                d_vars.append(var)
            elif ("downconv" not in var.name) and ("separator" in var.name):
                g_vars.append(var)
        for x in d_vars:
            assert x not in g_vars
        for x in g_vars:
            assert x not in d_vars
        print("    [*] Number of Trainable discriminator variables: {}".format(Utils.getNumParams(d_vars)))
        print("    [*] Number of Trainable generator variables: {}".format(Utils.getNumParams(g_vars)))

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    with tf.variable_scope("separator_solver"):
        separator_solver = tf.train.AdamOptimizer(learning_rate=model_config["init_sup_sep_lr"]).minimize(separator_loss, var_list=separator_vars)
    d_solver = None
    g_solver = None
    if model_config["discriminated"]:
        with tf.variable_scope("d_solver"):
            # Use RMSprop ?
            d_solver = tf.train.AdamOptimizer(learning_rate=model_config["d_init_sup_sep_lr"]).minimize(d_loss, var_list=d_vars)
        with tf.variable_scope("g_solver"):
            g_solver = tf.train.AdamOptimizer(learning_rate=model_config["g_init_sup_sep_lr"]).minimize(g_adv_loss, var_list=g_vars)


    # SUMMARIES
    if model_config["discriminated"]:
        tf.summary.scalar("g_adv_loss", g_adv_loss, collections=["gen", "unsup"])
        tf.summary.scalar("d_rl_loss", d_rl_loss, collections=["disc", "unsup"])
        tf.summary.scalar("d_fk_loss", d_fk_loss, collections=["disc", "unsup"])
    tf.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    sup_summaries = tf.summary.merge_all(key='sup')
    
    disc_summaries = tf.summary.merge_all(key='disc') if model_config["discriminated"] else None
    gen_summaries = tf.summary.merge_all(key='gen') if model_config["discriminated"] else None

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
        if os.path.exists(load_model) and os.path.isdir(load_model):
            restorer.restore(sess, tf.train.latest_checkpoint(load_model))
        else:
            restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    saver = tf.train.Saver(tf.global_variables(),
                                     write_version=tf.train.SaverDef.V2)

    if not model_config["discriminated"]:
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

    else:
        _global_step = sess.run(global_step)
        _d_global_step = sess.run(d_global_step)
        _g_global_step = sess.run(g_global_step)
        for _ in range(model_config["epoch_it"]):
            for _ in range(model_config["d_epoch_it"]):
                _, _disc_summaries = sess.run([d_solver, disc_summaries])
                writer.add_summary(_disc_summaries, global_step=_d_global_step)
                if _d_global_step % 100 == 0:
                    print('    [{}] Current step: {}'.format("disc", _d_global_step))
                _d_global_step = sess.run(increment_d_global_step)
            
            for _ in range(model_config["g_epoch_it"]):
                _, _gen_summaries = sess.run([g_solver, gen_summaries])
                writer.add_summary(_gen_summaries, global_step=_g_global_step)
                if _g_global_step % 100 == 0:
                    print('    [{}] Current step: {}'.format("gen", _g_global_step))
                _g_global_step = sess.run(increment_g_global_step)
                # TRAIN SEPARATOR
            _, _sup_summaries = sess.run([separator_solver, sup_summaries])
            writer.add_summary(_sup_summaries, global_step=_global_step) 
            if _global_step % 100 == 0:
                print('    [{}] Current step: {}'.format("sep", _global_step))
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
    
    d_epoch_list = [0,0]
    g_epoch_list = [0,0]
    epoch_list = [2000,2000]
    if model_config["discriminated"]:
        d_epoch_list = [0,0,50,5,2]
        g_epoch_list = [0,0,1,1,1]
        epoch_list = [2000,2000,2000,2000,2000]
    for i in range(len(epoch_list)):
        worse_epochs = 0
        if i>=1:
            if i==1:
                print("Finished first round of training, now entering fine-tuning stage")
                model_config["batch_size"] *= 2
                model_config["init_sup_sep_lr"] = model_config["init_sup_sep_lr"] / 10
 
        model_config["epoch_it"] = epoch_list[i]
        model_config["d_epoch_it"] = d_epoch_list[i]
        model_config["g_epoch_it"] = g_epoch_list[i]
            
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
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Optimize in a supervised fashion until validation loss worsens
    model_path = None
    #model_path = './checkpoints/unet_9_normal_frozen_downsample_False-237378'
    #model_path = './checkpoints/unet_9_normal_frozen_downsample_False-274777'
    #model_path = './checkpoints/unet_9_normal_frozen_downsample_False-375889'
    #model_path = './checkpoints/unet_9_normal_frozen_downsample_False-291214'
    
    #not imp model_path = './checkpoints/unet_9_normal_frozen_downsample_False-389011'
    #model_path = './checkpoints/unet++_9_normal_frozen_downsample_False-485470'
    #model_path = './checkpoints/unet++_9_deep_supervised_frozen_downsample_False-869415/'
    
    sup_model_path, sup_loss = optimise(model_path=model_path)
    print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))

    # Evaluate trained model on MUSDB
    # TODO
    print(model_config["estimates_path"])
    Evaluate.produce_musdb_source_estimates(model_config, sup_model_path, model_config["musdb_path"], model_config["estimates_path"])
