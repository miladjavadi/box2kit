from box2kit.autoconcat import gen_codebook as train_match
from box2kit.autoconcat import transfer as test_match
from box2kit.affine import train as train_affine
# from box2kit.affine import transfer as test_affine
from box2kit.gantransfer import train as train_lsae
from box2kit.gantransfer import transfer as test_lsae
from box2kit.svae import train_gan as train_w2v
# from box2kit.svae import transfer as test_w2v

import yaml
import argparse

def train_models(config, scenario):
    scripts = {"match": train_match,
               "affine": train_affine,
               "lsae": train_lsae,
               "w2v": train_w2v}
    
    for model in config["models"]:
        training_script = scripts[model["type"]]

        args_dict = model["args"]
        args_dict["target"] = scenario["target"]
        args_dict["output"] = scenario["output"]
        args = argparse.Namespace(**args_dict)

        training_script.main(args)

def main():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    scenarios = config["scenarios"]

    # scenarios = [{"target": "training_data/808_long", "output": "training_data/funk_long"},
    #              {"target": "training_data/808_short", "output": "training_data/funk_short"},
    #              {"target": "training_data/beatbox", "output": "training_data/drum_kit"}]
    
    for scenario in scenarios:
        train_models(config, scenario)

    
if __name__ == "__main__":
    main()