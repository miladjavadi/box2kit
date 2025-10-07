# box2kit: A Demonstration of Percussion Instrument Timbre Transfer Using DAC-Encoded Recordings

box2kit is a collection of models designed to transform the timbre of target percussion instrument recordings to that of another instrument.

The idea for this project came from me wanting bespoke, natural-sounding acoustic drum tracks for my own songs, but having neither access to a drum kit, nor the patience to meticulously program tracks using a VST.
I spend a majority of my songwriting process in the voice memo app on my phone, where I usually beatbox placeholders for actual drum tracks.
I figured that a fun, intuitive way to approach this problem could be through a piece of software which turns those beatboxed grooves into realistic-sounding acoustic drums. Hence, the name, *box2kit*.

## Background

## Model Types

## Usage
Each of the four model types are implemented as submodules inside the ```box2kit``` module.

### Dataset Management
By default, dataset directories are expected to contain one folder for training data, and an optional folder for validation data.
These folders should be named ```training``` and ```validation```, respectively.
Each folder should contain two subfolders of **paired** target and output instrument recordings, by default named ```target``` and ```output```.
The recordings should be named such that each target recording ```.../target/x.wav``` has a symbolically identical output recording ```.../output/x.wav```.

By default, the dataset file tree should look similar to this:
```
└── data/
    ├── training/
    │   ├── target/
    │   │   ├── a.wav
    │   │   ├── b.wav
    │   │   ├── c.wav
    │   │   └── ...
    │   └── output/
    │       ├── a.wav
    │       ├── b.wav
    │       ├── c.wav
    │       └── ...
    └── validation/
        ├── target/
        │   ├── x.wav
        │   ├── y.wav
        │   ├── z.wav
        │   └── ...
        └── output/
            ├── x.wav
            ├── y.wav
            ├── z.wav
            └── ...
```

For easier file management, other subsets of data, such as testing data, can also be included in the same directory, and will be ignored during training.
The training scripts only look for ```.wav```-files, and will ignore files with other extensions.

The preferred structure of the dataset directory can be configured in ```box2kit/configs/.../global.yaml```.

### Model Training
All models follow the same naming/syntax conventions.
A model can be trained by calling the ```train``` script inside their respective submodule:
```bash
python -m box2kit.<model_type>.train path/to/data --name experiment_name --test path/to/test_file 
```
```--name``` sets the name of the checkpoint folder/model file for the model within the respective model type's logs folder.
```--test``` (only available for model types ```neural``` and ```waveenc```) lets you provide an audio file to occasionally test your model during training.
The available model types are ```neural```, ```match```, ```affine```, and ```waveenc```.

### Model Configuration
Several hyperparameters for model structure and training are configurable from the various configuration files found in ```box2kit/configs/```.

A default set of config files ```box2kit/configs/default``` is provided. However, modifications to these files may be overwritten upon re-pulling the repo. If you wish to make modifications to the model configurations, it is therefore recommended to first copy the ```default``` folder as follows:
```
user@puter:<project_dir>/box2kit/configs$ cp -r default user 
```
and make any modifications to the ```user```-contained configuration files.

### Training from Checkpoint
Training can be resumed from an existing ```neural``` or ```waveenc``` checkpoint by passing it to the ```train``` script using the ```--ckpt``` flag:
```
user@puter:<project_dir>$ python -m box2kit.<model_type>.train path/to/data --name experiment_name --ckpt models/.../checkpoint.ckpt
```
Logs from the resumed model will be saved as a new version under the specified experiment name.

## Timbre Transfer
Similar to for training, all model type submodules contain a respective ```transfer``` script, which follows a common syntax:
```
python -m box2kit.<model_type>.transfer checkpoint/path.ckpt input/files --output output/files
```
Input files must be contained in a folder, pointed to by the second positional argument.
```--output``` specifies the name of the output file directory.

For ```match``` models, ```--order``` can be used to specify the order of the match search transfer algorithm.
For ```neural``` and ```affine``` models, ```--requantize``` can be used to requantize the generated DAC-latents using DAC's RVQ.