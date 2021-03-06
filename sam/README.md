## Sam's Experiments

This folder is where I put my scripts for training experiments which are a bit more involved than being runnable in a Jupyter notebook.
Much of the training loop/data loading code was inspired by Kally's notebooks, but I have since modified it quite a lot.
As a result, all setup instructions are deferred to Kally's README.

Note: as per Kally's README, you must have an environment variable `PATH_TO_GAT` pointing to the folder where this code can `import gat`.

### Setup using Docker

This section is optional, and contains instructions on how to run these scripts inside a Docker environment.
Note: in order to use the GPU you probably want to be using `nvidia-docker`.
Firstly, to build the Docker image, navigate to the root of the repository and do:
```nvidia-docker build -t atml_img -f sam/Dockerfile .```
which creates an image called `atml_img`.
Now, to create a container, you can do:
```nvidia-docker run --rm -it --mount type=bind,src=/home/<MY-USERNAME>/atml-group-11,dst=/atml-group-11 --user $(id -u):$(id -g) --name my_atml atml_img```
where `<MY-USERNAME>` should be replaced by your username, assuming the repo `atml-group-11` is in your home directory (this `--mount` option just allows the container to see the repository).

### Models and Layers

All of my models and layers have been implemented alongside Kally's in `kally/src/gat`.

### Running Experiments

All experiments are run from a script starting with the `run.py` script.
You need to give this script (at least) two arguments.
The first is the name of the dataset to train on (currently, transductive experiments only support `cora` and inductive datasets only support `ppi`).
Then, you need to give it one or more _configuration filenames_ (for examples on the format, see the `configs` folder).
These can include `*` and `**` to match arbitrary string patterns and arbitrary directories, respectively.

Each configuration file defines a model type with several parameters, but it also defines a "tag".
This tag is to determine what this model is in competition with (in terms of maximising performance on the validation set).
For example, if you wanted to train three different transformers with different dropout rates to see which is best, you would give them all the same tag, and then at the end of the program the model will print whichever it found to be best.
_All_ of them will be saved, however.

The model files, as well as their training logs, will always be written next to the configurations, with the same file name but a different extension.
Training logs contain the JSON model configuration, followed by CSV-formatted data with the following data, one per epoch:
```model-index,epoch,loss,val_acc,test_acc,is_improvement,p_att,p_hid```
which represent as follows:
- The model index, in case of repeated experiments. Currently this is not supported and is always 0.
- The epoch number
- The training loss at that epoch
- The validation set accuracy at that epoch
- The test set accuracy at that epoch
- Whether or not this validation set accuracy was an improvement on the previous best _for this model architecture_
- The dropout value on attention, if using annealed attention dropout
- The dropout value on the hidden layer, if using annealed hidden dropout

#### Parameter Sweeping

With one configuration file per model type, it might at first seem difficult/tedious to test a bunch of different model configurations, when trying to search for the right hyperparameters.
The `sweep.py` script makes it easy to specify one configuration file to do a grid search over, and automatically generates a bunch of config files for each parameter configuration.
**Please do not commit the output of this script to the repo!**
For example, to specify a grid seach over several parameters you could do:
```
{
    "tag" : "universal-transformer",
    "type" : "universal",
    "train_cfg" :
    {
        "learning_rate" : [5.0e-5, 5.0e-4, 5.0e-3],
        "weight_decay" : [3.0e-3, 3.0e-4, 0.0],
        "max_epoch" : 100,

        "step_lr" :
        {
            "step" : [50, 25],
            "gamma" : 0.8
        }
    },
    "model_kwargs" :
    {
        "internal_dim" : 64,
        "num_layers" : [2, 4],
        "num_heads" : 8,
        "identity_bias" : [0.0, 0.01, 0.02],
        "dropout_hidden" : [0.01, 0.0, 0.1]
    }
}
```
then the corresponding sweep command would be
```python sweep.py config-patterns/ut-small.json models/ train_cfg.learning_rate train_cfg.weight_decay train_cfg.step_lr.step model_kwargs.num_layers model_kwargs.identity_bias model_kwargs.dropout_hidden```

#### Printing Results

Each configuration file during training produces two things (i) a training log, and (ii) the weights of the best epoch of that model's training.
To calculate, therefore, which model configuration is best, you need to look at all of these training logs.
A script which does this in an automated way is the `results_agg.py` script.

The `results_agg.py` script contains a few options to make the output cleaner.
For example, `--configs` only prints the JSON configs which performed best, and `--filenames` only prints the filenames of the corresponding training logs of the best performers.
The latter is useful because it means you can execute
```python results_agg.py --filenames "models/*.log" | xargs -I {} sh -c 'filename=$1; allfile=${filename%.log}; cp $allfile.* best-models/' -- {}```
which copies not only the training log, but the config file and model file, of the best models in the `models/` folder to the `best-models/` folder.

### Results

The chosen architectures, training results and models are stored in the `best-models/` folder.
