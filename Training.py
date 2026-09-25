from sacred import Experiment
from sacred import SETTINGS
from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os
import functools

import Datasets
import Utils
import Models.UnetSpectrogramSeparator
import Models.UnetAudioSeparator
import Models.NestedUnetSpectrogramSeparator
import Models.NestedUnetAudioSeparator
import Test
import Evaluate

SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False

ex = Experiment('Nested WaveUnet Training', ingredients=[config_ingredient])

@ex.config
# Executed for training, sets the seed value to the Sacred config so that Sacred fixes the Python and Numpy RNG to the same state everytime.
def set_seed():
    seed = 1337

def compute_loss(model_config, separator_sources, batch):
    separator_loss = 0.0
    for key in model_config["source_names"]:
        real_source = batch[key]
        sep_source = separator_sources

        if model_config["network"] in ("unet_spectrogram", "unet++_spectrogram") and not model_config["raw_audio_loss"]:
            window = functools.partial(tf.signal.hann_window, periodic=True)
            stfts = tf.signal.stft(
                tf.squeeze(real_source, 2),
                frame_length=1024,
                frame_step=768,
                fft_length=1024,
                window_fn=window
            )
            real_mag = tf.abs(stfts)
            sub_separator_loss = 0.0
            if model_config["deep_supervised"]:
                for i in range(model_config["min_sub_num_layers"], model_config["num_layers"]):
                    sub_separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source[i][key]))
                sub_separator_loss /= float(model_config["num_layers"])
                separator_loss += sub_separator_loss
            else:
                separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source[key]))
        else:
            sub_separator_loss = 0.0
            if model_config["deep_supervised"]:
                for i in range(model_config["min_sub_num_layers"], model_config["num_layers"]):
                    sub_separator_loss += tf.reduce_mean(tf.square(real_source - sep_source[i][key]))
                separator_loss += sub_separator_loss / float(model_config["num_layers"])
            else:
                separator_loss += tf.reduce_mean(tf.square(real_source - sep_source[key]))
    # Normalise by number of sources
    separator_loss = separator_loss / float(model_config["num_sources"])
    return separator_loss

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

    # Build model by running dummy forward pass
    dummy_input = tf.zeros(sep_input_shape, dtype=tf.float32)
    _ = separator_class(dummy_input, training=False, return_spectrogram=not model_config["raw_audio_loss"])

    dataset = Datasets.get_dataset(model_config,
                                   sep_input_shape,
                                   sep_output_shape,
                                   partition="train")
    data_iter = iter(dataset)

    print("Training...")

    # Set up optimizer and checkpointing
    step_var = tf.Variable(0, trainable=False, dtype=tf.int64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config["init_sup_sep_lr"])

    ckpt = tf.train.Checkpoint(model=separator_class, optimizer=optimizer, step=step_var)
    ckpt_dir = os.path.join(model_config["model_base_dir"], model_config["experiment_id"], str(experiment_id))
    os.makedirs(ckpt_dir, exist_ok=True)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)

    if load_model is not None:
        Utils.load_model_checkpoint(separator_class, load_model, optimizer=optimizer)

    log_dir = os.path.join(model_config["log_dir"], model_config["experiment_id"])
    os.makedirs(log_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(log_dir)

    print("Sep_Vars: " + str(Utils.getNumParams(separator_class.trainable_variables)))
    print("Num of variables: " + str(len(separator_class.variables)))

    @tf.function
    def train_step(batch_data):
        with tf.GradientTape() as tape:
            separator_sources = separator_class(
                batch_data["mix"],
                training=True,
                return_spectrogram=not model_config["raw_audio_loss"]
            )
            loss = compute_loss(model_config, separator_sources, batch_data)
        grads = tape.gradient(loss, separator_class.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, separator_class.trainable_variables) if g is not None]
        optimizer.apply_gradients(grads_and_vars)
        return loss

    # Start training loop
    for _ in range(model_config["epoch_it"]):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            batch = next(data_iter)

        curr_loss = train_step(batch)
        step_var.assign_add(1)
        _global_step = int(step_var.numpy())

        with writer.as_default():
            tf.summary.scalar("sep_loss", curr_loss, step=_global_step)

        if _global_step % 100 == 0:
            print('    [{}] Current step: {} Loss: {:.6f}'.format(model_config["network"], _global_step, float(curr_loss.numpy())))

    # Epoch finished - Save model
    print("Finished epoch!")
    save_path = manager.save(checkpoint_number=_global_step)
    writer.flush()
    writer.close()

    return save_path

@config_ingredient.capture
def optimise(model_config, experiment_id, model_path=None):
    epoch = 0
    best_loss = 10000.0
    best_model_path = model_path
    curr_lr = model_config["init_sup_sep_lr"]
    for i in range(3):
        worse_epochs = 0
        if i >= 1:
            print("Finished first round of training, now entering fine-tuning stage")
            if i == 3:
                model_config["batch_size"] *= 2
            model_config["init_sup_sep_lr"] = curr_lr
        curr_lr /= 10.0
        while worse_epochs < model_config["worse_epochs"]:
            # Early stopping on validation set after a few epochs
            print("EPOCH: " + str(epoch))
            model_path = train(load_model=model_path)
            curr_loss = Test.test(model_config,
                                  model_folder=os.path.join(model_config["experiment_id"], str(experiment_id)),
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
    print("TRAINING FINISHED - TESTING WITH BEST MODEL " + str(best_model_path))
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
    for d in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(d):
            os.makedirs(d)

    sup_model_path = "./checkpoints/unet++_12_normal-599740/599740-110000"
    Evaluate.produce_musdb_source_estimates(model_config, sup_model_path, model_config["musdb_path"], model_config["estimates_path"])
