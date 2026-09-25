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

def test(model_config, partition, model_folder, load_model):
    # Determine input and output shapes
    # param: disc_input_shape: shape of test/discriminator input
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]

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
    assert ((sep_input_shape[1] - sep_output_shape[1]) % 2 == 0)

    # Build model by running dummy forward pass
    dummy_input = tf.zeros(sep_input_shape, dtype=tf.float32)
    _ = separator_class(dummy_input, training=False, return_spectrogram=not model_config["raw_audio_loss"])

    if load_model is not None:
        Utils.load_model_checkpoint(separator_class, load_model)

    log_path = os.path.join(model_config["log_dir"], model_folder)
    os.makedirs(log_path, exist_ok=True)
    writer = tf.summary.create_file_writer(log_path)

    # Creating the batch generators
    dataset = Datasets.get_dataset(model_config, 
                                   sep_input_shape, 
                                   sep_output_shape, 
                                   partition=partition)

    print("Testing...")

    total_loss = 0.0
    batch_num = 0

    for batch in dataset:
        separator_sources = separator_class(
            batch["mix"],
            training=False,
            return_spectrogram=not model_config["raw_audio_loss"]
        )

        batch_loss = 0.0
        for key in model_config["source_names"]:
            real_source = batch[key]
            sep_source = separator_sources[-1][key] if model_config["deep_supervised"] else separator_sources[key]

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
                batch_loss += tf.reduce_mean(tf.abs(real_mag - sep_source))
            else:
                batch_loss += tf.reduce_mean(tf.square(real_source - sep_source))

        batch_loss = batch_loss / float(model_config["num_sources"])
        total_loss += float(batch_loss.numpy())
        batch_num += 1

    mean_loss = total_loss / float(max(batch_num, 1))

    with writer.as_default():
        tf.summary.scalar("test_loss", mean_loss, step=0)

    writer.flush()
    writer.close()

    print("Finished testing - Mean MSE: " + str(mean_loss))
    return mean_loss
