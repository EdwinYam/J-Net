import os
import numpy as np
from sacred import Ingredient

config_ingredient = Ingredient("cfg")

'''
This configuration is for experiment on server 508, aimed at original unet setting
'''
@config_ingredient.config
def cfg():
    # Base configuration
    model_config = {"musdb_path" : "/media/Datasets/musdb18/", # SET MUSDB PATH HERE, AND SET CCMIXTER PATH IN CCMixter.xml
                    "noisy_VCTK_path" : "/media/Datasets/noisy_VCTK/",
                    # "estimates_path" : "/media/WaveUnet/Source_Estimates/", # SET THIS PATH TO WHERE YOU WANT SOURCE ESTIMATES PRODUCED BY THE TRAINED MODEL TO BE SAVED. Folder itself must exist!
                    "data_path" : "data", # Set this to where the preprocessed dataset should be saved

                    "model_base_dir" : "checkpoints", # Base folder for model checkpoints
                    "log_dir" : "logs", # Base folder for logs files
                    "batch_size" : 16, # Batch size
                    "init_sup_sep_lr" : 1e-4, # Supervised separator learning rate
                    "d_init_sup_sep_lr" : 1e-4,
                    "g_init_sup_sep_lr" : 1e-4,
                    "epoch_it" : 2000, # Number of supervised separator steps per epoch
                    'cache_size': 1000, # Number of audio snippets buffered in the random shuffle queue. Larger is better, since workers put multiple examples of one song into this queue. The number of different songs that is sampled from with each batch equals cache_size / num_snippets_per_track. Set as high as your RAM allows.
                    'num_workers' : 6, # Number of processes used for each TF map operation used when loading the dataset
                    "num_snippets_per_track" : 100, # Number of snippets that should be extracted from each song at a time after loading it. Higher values make data loading faster, but can reduce the batches song diversity
                    'num_layers' : 12, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'input_filter_size' : 15, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'output_filter_size': 1, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'num_increase_filters': 24, # Number of filters added for convolution after first layer of network 
                    'num_initial_filters' : 24, # Number of filters for convolution in first layer of network
                    "num_frames": 16384, # DESIRED number of time frames in the output waveform per samples (could be changed when using valid padding)
                    'expected_sr': 22050,  # Downsample all audio input to this sampling rate
                    'mono_downmix': True,  # Whether to downsample the audio input
                    'output_type' : 'direct', # Type of output layer, either "direct" or "difference". Direct output: Each source is result of tanh activation and independent. DIfference: Last source output is equal to mixture input - sum(all other sources)
                    'output_activation' : 'tanh', # Activation function for output layer. "tanh" or "linear". Linear output involves clipping to [-1,1] at test time, and might be more stable than tanh
                    'context' : False, # Type of padding for convolutions in separator. If False, feature maps double or half in dimensions after each convolution, and convolutions are padded with zeros ("same" padding). If True, convolution is only performed on the available mixture input, thus the output is smaller than the input
                    'network' : 'unet', # Type of network architecture, either unet (our model) or unet_spectrogram (Jansson et al 2017 model)
                    'upsampling' : 'linear', # Type of technique used for upsampling the feature maps in a unet architecture, either 'linear' interpolation or 'learned' filling in of extra samples
                    'task' : 'voice', # Type of separation task. 'voice' : Separate music into voice and accompaniment. 'multi_instrument': Separate music into guitar, bass, vocals, drums and other (Sisec)
                    'augmentation' : True, # Random attenuation of source signals to improve generalisation performance (data augmentation)
                    'raw_audio_loss' : True, # Only active for unet_spectrogram network. True: L2 loss on audio. False: L1 loss on spectrogram magnitudes for training and validation and test loss
                    'worse_epochs' : 15, # Patience for early stoppping on validation set
                    #'deep_supervised': False, # Whether to use nested arch.
                    #'min_sub_num_layers': 0,
                    #'sub_num_layers': None,
                    'min_skip_num_layers': 0,
                    #'d_min_sub_num_layers': 0,
                    #'evaluate_subnet': False,
                    'add_random_layer': False,
                    'use_meanvar': False,
                    'random_downsample_layer': list(),
                    'random_upsample_layer': list(),
                    'remove_random': False,
                    #'discriminated': False,
                    #'g_loss_weight': 1.0,
                    #'d_epoch_it': 1,
                    #'g_epoch_it': 1,
                    'semi_supervised': False,
                    'semi_loss_ratio': 0.1,
                    #'residual': False,
                    'random_recovery': False,
                    'experiment_id': ''
                    }
    experiment_id = np.random.randint(0,1000000)

    # Set output sources
    if model_config["task"] == "multi_instrument":
        model_config["source_names"] = ["bass", "drums", "other", "vocals"]
    elif model_config["task"] == "voice":
        model_config["source_names"] = ["accompaniment", "vocals"]
    elif model_config["task"] == "denoise":
        model_config["source_names"] = ["clean"]
    else:
        raise NotImplementedError
    model_config["num_sources"] = len(model_config["source_names"])
    model_config["num_channels"] = 1 if model_config["mono_downmix"] else 2
    # model_config["loss_type"] = "deep_supervised" if model_config["deep_supervised"] else "normal" 
    model_config["experiment_id"] = "{}-{}_{}_{}-{}".format(model_config["network"], model_config["num_layers"], 'RNGlayer', model_config["add_random_layer"], experiment_id)
    model_config["estimates_path"] = os.path.join("/media/WaveUnet/Source_Estimates", model_config["experiment_id"])

@config_ingredient.named_config
def baseline():
    print("Training baseline model")

@config_ingredient.named_config
def baseline_diff():
    print("Training baseline model with difference output")
    model_config = {
        "output_type" : "difference"
    }

@config_ingredient.named_config
def baseline_context():
    print("Training baseline model with difference output and input context (valid convolutions)")
    model_config = {
        "output_type" : "difference",
        "context" : True
    }

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# This is config for baseline
@config_ingredient.named_config
def baseline_10():
    print("Train model with fix random encoder/downsampling part")
    model_config = {
        "batch_size": 16,
        "num_layers": 10,
        "num_increase_filters": 24,
        "network": "unet",
        "output_type": "difference",
        "context": False,
        "mono_downmix": False,
        "add_random_layer": False,
        "use_meanvar": False,
        "random_downsample_layer": list()
        "random_upsample_layer": list()
        "min_skip_num_layers": 0,
        "remove_random": False,
    }

@config_ingredient.named_config
def baseline_12():
    print("Train model with fix random encoder/downsampling part")
    model_config = {
        "batch_size": 16,
        "num_layers": 12,
        "num_increase_filters": 24,
        "network": "unet",
        "output_type": "difference",
        "context": True,
        "mono_downmix": False,
        "add_random_layer": False,
        "use_meanvar": False,
        "random_downsample_layer": list()
        "random_upsample_layer": list()
        "min_skip_num_layers": 0,
        "remove_random": False,
    }

# This is config for random fix encoder 
@config_ingredient.named_config
def fix_downsample():
    print("Train model with fix random encoder/downsampling part")
    model_config = {
        "batch_size": 16,
        "num_layers": 10,
        "num_increase_filters": 24,
        "network": "unet",
        "output_type": "difference",
        "context": False,
        "mono_downmix": False,
        "add_random_layer": True,
        "use_meanvar": False,
        "random_downsample_layer": [ i for i in range(10) ]
        "random_upsample_layer": list()
        "min_skip_num_layers": 0,
        "remove_random": False,
    }
    
#=====================================================#
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#

# This is config for current repo
@config_ingredient.named_config
def nested_context():
    print("Train nested model with direct output and input context (valid convs.)")
    model_config = {
        "batch_size": 8,
        "num_layers": 10,
        "num_increase_filters": 16,
        "network": "unet++",
        "output_type": "direct",
        "context": True,
        "mono_downmix": False,
        "deep_supervised": True,
    }

@config_ingredient.named_config
def nested_diff():
    print("Train nested model with difference output and input context")
    model_config = {
        "batch_size": 8,
        "num_layers": 12,
        "num_increase_filters": 24,
        "network": "unet++",
        "output_type": "difference",
        "context": True,
        "mono_downmix": False,
        "deep_supervised": True,
    }

# According to the original paper, this is the best setting
@config_ingredient.named_config
def baseline_stereo():
    print("Training baseline model with difference output and input context (valid convolutions) and stereo input/output")
    model_config = {
        #"init_sup_sep_lr": 1e-5,
        "num_layers": 10,
        "output_type" : "difference",
        "context" : True,
        "mono_downmix" : False,
        "frozen_downsample": True,
        "min_skip_num_layers": 4
    }


@config_ingredient.named_config
def baseline_stereo_direct():
    print("Training baseline model with difference output and input context (valid convolutions) and stereo input/output")
    model_config = {
        #"init_sup_sep_lr": 1e-5,
        "num_layers": 10,
        "output_type" : "direct",
        "context" : True,
        "mono_downmix" : False,
        "residual" : True,
        "frozen_downsample": False,
        "min_skip_num_layers": 0
    }
#===================================================#
#####################################################
@config_ingredient.named_config
def baseline_direct():
    model_config = {
        "num_layers": 9,
        "output_type": "direct",
        "context": False,
        "mono_downmix": False,
        "residual": False,
        "frozen_downsample": False,
        "min_skip_num_layers": 0,
        "random_recovery": False
    }

@config_ingredient.named_config
def baseline_residual():
    model_config = {
        "num_layers": 9,
        "output_type": "direct",
        "context": False,
        "mono_downmix": False,
        "residual": True,
        "frozen_downsample": False,
        "min_skip_num_layers": 0,
        "random_recovery": False
    }


@config_ingredient.named_config
def baseline_recovery():
    model_config = {
        "num_layers": 9,
        "output_type": "direct",
        "context": False,
        "mono_downmix": False,
        "residual": False,
        "frozen_downsample": False,
        "min_skip_num_layers": 0,
        "random_recovery": True
    }

@config_ingredient.named_config
def baseline_semi_supervised():
    model_config = {
        "num_layers": 9,
        "output_type": "direct",
        "context": False,
        "mono_downmix": False,
        "residual": False,
        "frozen_downsample": False,
        "min_skip_num_layers": 0,
        "random_recovery": False,
        "semi_supervised": True
    }

@config_ingredient.named_config
def baseline_discriminated():
    model_config = {
        "num_layers": 9,
        "output_type": "direct",
        "context": False,
        "mono_downmix": False,
        "residual": False,
        "frozen_downsample": False,
        "min_skip_num_layers": 0,
        "random_recovery": False,
        "semi_supervised": False,
        "discriminated": True,
    }


@config_ingredient.named_config
def full():
    print("Training full singing voice separation model, with difference output and input context (valid convolutions) and stereo input/output, and learned upsampling layer")
    model_config = {
        "output_type" : "difference",
        "context" : True,
        "upsampling": "learned",
        "mono_downmix" : False
    }

@config_ingredient.named_config
def full_44KHz():
    print("Training full singing voice separation model, with difference output and input context (valid convolutions) and stereo input/output, and learned upsampling layer, and 44.1 KHz sampling rate")
    model_config = {
        "output_type" : "direct",
        "context" : True,
        "upsampling": "learned",
        "mono_downmix" : False,
        "expected_sr" : 44100
    }

@config_ingredient.named_config
def baseline_context_smallfilter_deep():
    model_config = {
        "output_type": "direct",
        "context": True,
        "num_layers" : 14,
        "duration" : 7,
        "filter_size" : 5,
        "merge_filter_size" : 1
    }

@config_ingredient.named_config
def full_multi_instrument():
    print("Training multi-instrument separation with best model")
    model_config = {
        "output_type": "difference",
        "context": True,
        "upsampling": "linear",
        "mono_downmix": False,
        "task" : "multi_instrument"
    }

@config_ingredient.named_config
def baseline_comparison():
    model_config = {
        "batch_size": 4, # Less output since model is so big. Doesn't matter since the model's output is not dependent on its output or input size (only convolutions)

        "output_type": "difference",
        "context": True,
        "num_frames" : 768*127 + 1024,
        "duration" : 13,
        "expected_sr" : 8192,
        "num_initial_filters" : 34
    }

@config_ingredient.named_config
def unet_spectrogram():
    model_config = {
        "batch_size": 4, # Less output since model is so big.

        "network" : "unet_spectrogram",
        "num_layers" : 6,
        "expected_sr" : 8192,
        "num_frames" : 768 * 127 + 1024, # hop_size * (time_frames_of_spectrogram_input - 1) + fft_length
        "duration" : 13,
        "num_initial_filters" : 16
    }

@config_ingredient.named_config
def unet_spectrogram_l1():
    model_config = {
        "batch_size": 4, # Less output since model is so big.

        "network" : "unet_spectrogram",
        "num_layers" : 6,
        "expected_sr" : 8192,
        "num_frames" : 768 * 127 + 1024, # hop_size * (time_frames_of_spectrogram_input - 1) + fft_length
        "duration" : 13,
        "num_initial_filters" : 16,
        "raw_audio_loss" : False
    }
