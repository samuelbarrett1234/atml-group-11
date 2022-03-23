## Guide for this part of the project: 

### 1. Source code for the models

The models are located in `src/gat`.

The `layers` module contains the MultiHeadAttention layer from the GAT paper and the `models` module contains the Transductive and Inductive variants. 
These files have both been extended with 'vanilla' transformer variants.

I have tried to follow all the details described in the GAT paper (e.g. dropout in the Transductive model and skip-connection accross the intermediate layer in the Inductive model)

To import and use the models in your own code:

- construct a variable `PATH_TO_GAT` that contains the path to the `src` directory

- run `sys.path.append(PATH_TO_GAT)`

- `from gat import GAT_Inductive, GAT_Transductive, Layer_Attention_MultiHead_GAT` (or likewise for other models/layers).


### 2. Experiments (Kally)

I have prepared two demo notebooks - one running the Transductive model on the CORA dataset and one running the Inductive model on the PPI dataset. The notebooks are located in `src/test` 

Again, I have tried to replicate the details described in the paper as much as possible, the only explicit difference being the early stopping procedure - I don't use their 100-epoch tolerance, instead opting for 20-epoch tolerance in CORA and 10-epoch tolerance in PPI. 

The results achieved in the notebooks are:

- 0.873 accuracy on CORA - this seems to be an easier task computationally-wise, running smoothly on CPU without much delay. The weights of the model are available in the file `src/test/trans_model.pt`

- 0.766 micro f1-score on PPI - this is a harder task computationally and I only run training for at most 50 epochs. The weights of the model are available in the file `src/test/ind_model.pt`

### 4. Dependencies 

All dependencies needed to run this code are available in the `Pipfile`. To create a virtual environment and install the dependencies run `pipenv install` in this root directory.
