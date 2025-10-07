import os

def get_checkpoint_path(checkpoint_folder: str, key: str = "last", descending: bool = True) -> str:
    checkpoint_files = [file for file in os.listdir(f"{checkpoint_folder}/checkpoints") if file[-5:] == ".ckpt"]

    checkpoints = [dict([["name", name]] + [attribute.split("=") for attribute in name.split("-")]) for name in checkpoint_files]
    
    try:
        checkpoint_name = sorted(checkpoints, key = lambda d: d[key], reverse = descending)[0]["name"]
    
    except KeyError:
        raise KeyError(f'Key "{key}" not found in checkpoint file name.')
    
    checkpoint_path = f"{checkpoint_folder}/checkpoints/{checkpoint_name}"
    return checkpoint_path