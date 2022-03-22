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

Experiments are run from a script starting with `run_`; for example if you want to run transductive experiments you invoke `run_transductive.py`.
You need to give this script (at least) 3 arguments.
The first is the name of the dataset to train on (currently, transductive experiments only support `cora`.)
Then, you need to give it a CSV filename to log training data to (which will be appended to the end of the file).
Finally, you need to give it one or more _configuration filenames_ (for examples on the format, see the `configs` folder).
These can include `*` and `**` to match arbitrary string patterns and arbitrary directories, respectively.

Each configuration file defines a model type with several parameters, but it also defines an "output name".
This output name is to determine what this model is in competition with (in terms of maximising performance on the validation set).
For example, if you wanted to train three different transformers with different dropout rates to see which is best, you would give them all the same output name, so that only the best epoch for the best of the three models is saved.
The filename of the saved model is then taken from this output name.

### Results

