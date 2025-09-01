from box2kit.autoconcat import gen_codebook as train_match
from box2kit.autoconcat import transfer as test_match
from box2kit.affine import train as train_affine
# from box2kit.affine import transfer as test_affine
from box2kit.gantransfer import train as train_nested
from box2kit.gantransfer import transfer as test_nested
from box2kit.svae import train_gan as train_w2w
# from box2kit.svae import transfer as test_w2v
import box2kit.utils.load_data as uload

import yaml
import argparse
import os

def train_models(config, scenario, model_dir):
    scripts = {"match": train_match,
               "affine": train_affine,
               "nested": train_nested,
               "w2w": train_w2w,
               "w2w_self": train_w2w}
    
    for model in config["models"]:
        training_script = scripts[model["type"]]

        args_dict = model["args"]

        # for self-reconstructin w2w encoder (feature space proj), use output set as target set when training
        args_dict["target"] = scenario["target"] if model["type"] != "w2w_self" else scenario["output"]

        args_dict["output"] = scenario["output"]
        args_dict["tempo"] = scenario["tempo"]
        args_dict["name"] = uload.mkdir(f"{model_dir}/{scenario["name"]}/{model["type"]}")
        args = argparse.Namespace(**args_dict)

        training_script.main(args)

def main(args):
    config_path = args.config
    model_dir = args.modeldir

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    scenarios = config["scenarios"]

    # scenarios = [{"target": "training_data/808_long", "output": "training_data/funk_long"},
    #              {"target": "training_data/808_short", "output": "training_data/funk_short"},
    #              {"target": "training_data/beatbox", "output": "training_data/drum_kit"}]
    
    for scenario in scenarios:
        os.mkdir(f"{model_dir}/{scenario["name"]}")
        train_models(config, scenario, model_dir)

    
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="wwjd")
    parser.add_argument("--modeldir", help="Directory path to trained model batches.", metavar="path", type=str, default=uload.mkdir("model_batches"))
    parser.add_argument("--config", help="Path to config. file.", metavar="path", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args)