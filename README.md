# box2kit: A Demonstration of Percussion Instrument Timbre Transfer Using DAC-Encoded Recordings

box2kit is a collection of models designed to transform the timbre of percussion instrument recordings to that of another instrument.

The idea for this project came from me wanting bespoke, natural-sounding acoustic drum tracks for my own songs, but not having access to a drum kit, nor the patience to meticulously program tracks using a VST.
I spend a majority of my songwriting process in the voice memo app on my phone, where I usually beatbox placeholders for actual drum tracks.
I figured that a fun, intuitive way to approach this problem could be through a piece of software which turns those beatboxed grooves into realistic-sounding acoustic drums. Hence, the name, *box2kit*.

Timbre transfer is conventionally accomplished using reconstructive models, such as autoencoders or concatenative synthesizers, that have been trained on single-instrument recordings.
By passing recordings from a different *target instrument* to an already trained *output instrument* model, the internal features derived from those recordings are re-synthesized as if they were output instrument recordings.
What results are recordings that contain the same compositional and expressive information as the target instrument, but reproduced using the output instrument's timbre.

However, as shown in the [report attached to this project]() (not yet published), this approach may not work particularly well in all cases.
One such case is between percussion instruments with multiple distinct timbral "modes", for instance between beatboxing and acoustic drum kits.
A more effective solution in those cases could be to use models that are explicitly trained to transfer timbre between a target-output instrument pair.
Such models utilize *paired* target and output instrument recordings to derive timbral relationships between symbolically identical audio data.
These kinds of models are what are primarily explored in this project.

## Model Types

This repo contains four types of models for performing explicitly paired target-output instrument timbre transfer.

Three of these: *neural transfer*, *match search*, and *affine transformation*, are effectively vector transforms that are applied to latent space respresentations of audio signals, encoded using [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec).
There are two main reasons for this.
Firstly, by using DAC as an interface for our models, we effectively bypass the need for our models to learn feature representation, as this is already accomplished by DAC.
This potentially allows us to create sparser models, that are faster to train and less prone to overfitting when data is sparse, as is often the case for paired data modeling.
Secondly, DAC uses a learned Residual Vector Quantization (RVQ) quantization system to compress encoded signals.
This effectively restricts the output of the codec to "realistic"---or domain-relevant---frames of audio.
In turn, this may allow us to "correct" for slight inaccuracies in the output of our models.

A fourth model type, the *waveform autoencoder*, is a full waveform-to-waveform Variational Autoencoder (VAE).
It serves as a baseline against which to compare the DAC-interfacing models, and to demonstrate the concept of paired instrument timbre transfer on its own.

### Neural Transfer
Neural transfer models use Convolutional Neural Networks (CNNs) to transform target instrument latent vectors to symbolically equivalent output instrument latent vectors.
They are trained using a combination of waveform reconstruction loss, latent space distance, and adversarial loss.

### Match Search
Training match search models involves using a greedy algorithm to construct a codebook containing sequences of paired target instrument and output instrument latent vectors.
When new target recordings are passed, match search models segment them, and map each segment to their nearest neighboring target codebook entries in latent space.
Each segment is then replaced with the respective output codebook entries.

A $k$-order match search can be performed by matching each input segment with its $k$ nearest neighboring target codebook entries, and replacing it with the mean of their respective output codebook entries.
The output is then re-mapped to the nearest neighboring ouptut codebook entry.

### Affine Transformation
Affine transformation models use Random Sample Consensus (RANSAC) to derive an affine transform\
$$\hat{q}[i]=Ap[i] + b,$$\
which maps target latent vectors $p[i]$ to symbolically equivalent output vectors $\hat{q}[i]$.

### Waveform Autoencoder
The waveform autoencoder used in this project is mostly a recreation of the [Realtime Variational Autoencoder (RAVE)](https://github.com/acids-ircam/RAVE).
The main difference is that this implementation is adapted for cross-instrumental synthesis using paired training data, rather than identical reconstruction.

## Demonstration
[Generated example recordings featuring each model type can be found here.](https://miladjavadi.github.io/)

## Usage
The four model types presented [above](#model-types) are implemented as sub-packages inside the ```box2kit``` package.
Each sub-package contains the model declaration ```model.py```, a training script ```training.py```, and a post-training transfer script ```transfer.py```.
To ensure the correct working directory, the scripts should be called as submodules from the project's root folder:
```
user@puter:<project_dir>$ python -m box2kit.<model_type>.<script> 
```

### Dataset Management
Using default settings, dataset directories are expected to contain one folder for training data, and an optional folder for validation data.
These folders should be named ```training``` and ```validation```, respectively.
Each folder should contain two subfolders of **paired** target and output instrument recordings, by default named ```target``` and ```output```.
The recordings should be named such that each target recording ```.../target/x.wav``` has a symbolically identical output recording ```.../output/x.wav```.

With default settings, the dataset file tree should look similar to this:
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
A model can be trained by calling the ```train``` submodule inside their respective subpackage:
```
python -m box2kit.<model_type>.train path/to/data --name experiment_name --test path/to/test_file 
```
```--name``` sets the name of the checkpoint folder/model file for the model within the respective model type's logs folder.
```--test``` (only available for model types ```neural``` and ```waveenc```) lets you provide an audio file to occasionally test your model during training.
The available model types are ```neural```, ```match```, ```affine```, and ```waveenc```.

### Model Configuration
Several hyperparameters for model structure and training are configurable from the various configuration files found in ```box2kit/configs/```.

A default set of config files ```box2kit/configs/default``` is provided. However, modifications to these files may be overwritten upon re-pulling the repo.
If you wish to make modifications to the model configurations, it is therefore recommended to first copy the ```default``` folder as follows:
```
user@puter:<project_dir>/box2kit/configs$ cp -r default user 
```
and make any modifications to the ```user```-contained configuration files.

Configuration parameters can be temporarily overriden on execution of any ```train``` script, using ```--override```, together with a YAML-file containing keys and temporary replacement values.
For instance, ```override.yaml```, containing
```
neural/epochs: 500
global/training_target_path: "validation/output"
```
can be used to override to override the ```epochs``` parameter in ```neural.yaml```, and the ```training_target_path``` parameter in ```global.yaml```:
```
user@puter:<project_dir>$ python -m box2kit.neural.train path/to/data --override override.yaml
...
Overriding neural/epochs: 10000 -> 500
Overriding global/training_target_path: training/target -> validation/output
```
This can be useful for scheduling consecutive training of multiple models.

### Training from Checkpoint
Training can be resumed from an existing ```neural``` or ```waveenc``` checkpoint by passing it to the ```train``` script using the ```--ckpt``` flag:
```
user@puter:<project_dir>$ python -m box2kit.<model_type>.train path/to/data --name experiment_name --ckpt models/.../checkpoint.ckpt
```
Logs from the resumed model will be saved as a new version under the specified experiment name.

### Timbre Transfer
Similar to for training, all model type subpackages contain a respective ```transfer``` script, which follows a common syntax:
```
python -m box2kit.<model_type>.transfer checkpoint/path.ckpt input/files --out output/files
```
Input files must be contained in a folder, pointed to by the second positional argument.
```--out``` specifies the name of the output file directory.

For ```match``` models, ```--order``` can be used to specify the order of the match search transfer algorithm.
For ```neural``` and ```affine``` models, ```--rq``` can be used to requantize the generated DAC-latents using DAC's RVQ.
