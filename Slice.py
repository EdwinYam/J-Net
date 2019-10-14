import os
from sacred import Experiment
from sacred import SETTINGS
from Config import config_ingredient
import Evaluate

SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
ex = Experiment('Evaluate Sliced Nested WaveUnet', ingredients=[config_ingredient])

@ex.config
def set_seed():
    seed = 1337

@ex.automain
def run(cfg):
    sup_model_path = "./checkpoints/125802/125802-82000" 
    print("[Evaluation] Start evaluation ... ")
    model_config = cfg["model_config"]
    model_config["evaluate_subnet"] = True
    if model_config["network"] != "unet++":
        raise NotImplementedError

    print("[unet++] Start to Evaluate Sliced Nested Unet!")
    for i in range(model_config["num_layers"]-model_config["min_sub_num_layers"]):
        print(    "[unet++] Evaluating {}-layer subnet".format(i))
        model_config["estimates_path"] = os.path.join(model_confg["estimates_path"], str(i))
        model_config["sub_num_layers"] = i
        Evaluate.produce_musdb_source_estimates(model_config, sup_model_path, model_config["musdb_path"], model_config["estimates_path"])

