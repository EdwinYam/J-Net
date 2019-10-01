from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Nested WaveUnet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    model_path = os.path.join("checkpoints", "125802", "125802-82000") # Load stereo vocal model by default
    input_path = os.path.join("audio_examples", "mix_p232_280.wav") # Which audio file to separate
    output_path = None # Where to save results. Default: Same location as input.

@ex.automain
def main(cfg, model_path, input_path, output_path):
    model_config = cfg["model_config"]
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)
