- [moment-neural-network](#moment-neural-network)
  - [The architecture of this repository](#the-architecture-of-this-repository)
- [Getting Started](#getting-started)
  - [Quick start: three steps to run your first MNN model](#quick-start-three-steps-to-run-your-first-mnn-model)
  - [Custom your own MNN model](#custom-your-own-mnn-model)
  - [Configure the training options via argparser.](#configure-the-training-options-via-argparser)
  - [Run corresponding SNN simulation](#run-corresponding-snn-simulation)
- [Authors](#authors)
- [License](#license)

# moment-neural-network


## The architecture of this repository
* `mnn_core`: core modules implementing the moment activation and other building blocks of MNN.
* `models`: a module containging various network architectures for fast and convenient model construction
* `snn`: modules for reconstructing SNN from MNN and for simulating the corresponding SNN in a flexible manner.
* `utils`: a collection of useful utilities for training MNN (ANN compatible).

# Dependencies
(assume the user only has naked python)
Latest version of Python 3
pytorch
numpy
scipy
++

# Getting Started

## Quick start: three steps to run your first MNN model

1. Copy the example files, **./example/mnist/mnist.py** and **./example/mnist/mnist_config.yaml** to the root directory
2. Create two directory, **./checkpoint/** (for saving trained model results) and **./data/** (for downloading MNIST dataset).
3. Run the following command to call the script named `mnist.py` with the config file specified through the option:

    ```
    python mnist.py --config=./mnist.yaml
    ```

After training is finished, you should find four files in the **./checkpoint/mnist/** folder：

- two model files that contain the trained model parameters with extension .pth
- the config file used for running the training with extension .yaml
- one log file that records the model performance during the training with extension .txt
- one directroy called `mnn_net_snn_result` that stores the simulation result of the SNN reconstructed from the trained MNN

## Configure the MNN model

Let's review the content of **mnist.yaml**.
To specify the architecture of MNN, you can modify the `MODEL` section.
The settings under `meta` provide the information about model construction. 
Currently only mlp-like architecture is available (`arch: mnn_mlp`).
The setting `mlp_type` indicates the kind of mlp to be built. For `mnn_mlp`, the model contains one input layer, arbitrary number of hidden layers, and a linear decoder. 
You can change the widths of each layer by modifying the values under `structrue`. The setting `num_class` specifies the output dimension. 
See `mnn.models.mlp` for further details.

Next, The `CRITERION` section indicate the training criterion such as the loss function. 
The code will try to find the criterion from `source` that match the `name` and pass required `args` to it.
I have implemented a family of criteria for MNN, see `mnn_core.nn.criterion` for further details.

Similarly, the optimzer and data augmentation policy are defined under `OPTIMIZER` and `DATAAUG_TRAIN/VAL`, correspoding to the pytorch implementations (`torch.optim` and `torchvision.transforms` ).

Alternality, you can rewrite your own functions of `MnistTrainFuncs` in the script **mnist.py**.

There are some advanced options in the config file:
* `save_epoch_state`: at the start of each epoch, the code will store the model parameters.
* `input_prepare`: currently only *flatten_poisson* is valid. It means we first flatten input to a vector and regard it as independent Poisson rate code.
* `scale_factor`: only valid if `input_prepare` is *flatten_poisson*, used to control input range.
* `is_classify`: the task type, if `False`, the best model is determined by the epoch that has minimal loss.
* `background_noise`: this value will add to the diagonal of input covariance (Can be helpful if input covariance is very weak or close to singular)

## Configure additional training options via argparser.
I recommend you to read the func `deploy_config()` in `utils.training_tools.general_prepare`
Some important options:
* `seed`: fix the seed for all RNGs used by the model. By default it is `None` (not fixed)
* `bs`: batch size used in the data loader
* `dir`: directory name for saving training data
* `save_name`: the prefix of file name of training data
* `epochs`: the number of epochs to train.

How to use:
```
python main_script.py --config=./your_config_file.yaml --OPT=VALUE
```
**Note** all manual argument will be overwritten if the same keys are found in the provided **your_config_file.yaml**

## Run simulations of the reconstructed SNN
We provide utility to automatically reconstruct SNN based on the trained MNN. 
A custom simulator of SNN is provided with GPU support but you may use any SNN simulator of your choice.

## Customize your own MNN model
(How to add custom models to the MODEL folder; details of the model class)


# Lead authors

- **Zhichao Zhu** - *Chief architect and initial work* - [Zachary Zhu](https://github.com/Acturos)
- **Yang Qi** - *Algorithm design* [Yang Qi](https://github.com/qiyangku)

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details.

