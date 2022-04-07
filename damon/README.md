# Some Experiments with Graph Attention Networks

**Implementations and experiments corresponding to Sections 2.2 and 5.2 in the accompanying report**

*Contents of this directory by Damon Falck*

---

This directory contains implementations of various models based on GAT (from [Graph Attention Networks (2018)](https://arxiv.org/abs/1710.10903)) and GATv2 (from [How Attentive are Graph Attention Networks? (2022)](https://arxiv.org/abs/2105.14491)) and two sets of experiments involving them:

1. A reproduction of the results from the GAT paper, using exactly the same models and training procedures they used (this corresponds to Section 2.2 in the accompanying report);
2. A broader study comparing various variants of the GAT and GATv2 attention mechanisms on a wider array of datasets (corresponding to Section 5.2 in the report).

The results from these experiments can be found in the `notebooks` directory.

## Structure

The models and training are implemented in a Python package contained in `src/oxgat`. Within this:

- the `components` module defines individual GAT/GATv2 layers and attention heads;
- the `models` module defines trainable models according to the original paper;
- the `utils` module defines additional utilities.

The `notebooks` directory contains Jupyter notebooks running the two empirical studies mentioned above, with JSON results for the second study in `notebooks/results`.

HTML package documentation is available in the `docs` directory, and the reader is encouraged to look through this!

## Package installation

Notebooks here depend on the Python package `oxgat` mentioned above. This should be installed using
```
pip install -e src
```
from this directory; all dependencies will be automatically added.

## Implementation comments

My implementations are based on PyTorch Lightning, appropriately abstracting away details of the training process; a class heirarchy is defined in `models` which provides automated training interfaces for each model. In all cases I have provided *sparse tensor calculations* as built-in functionality, which substantially speeds up training.

I highly recommend running any of the experimental notebooks in a GPU-enabled environment; I have provided Colab links in the header of each notebook.

**Please see the accompanying report for a full discussion of the results from my experiments!**