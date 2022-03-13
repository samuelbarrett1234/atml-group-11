# Exploring Graph Attention Networks

**Kaloyan Aleksiev, Samuel Barrett, Damon Falck, Filip Mohov**

*Original implementation in this directory by Damon Falck*

This directory contains implementations of the experimental results in [Graph Attention Networks (2018)](https://arxiv.org/abs/1710.10903), as well as hopefully some extensions to be added soon.

## Status summary

Currently I have implemented the inductive and transductive GAT models defined in the original paper, with training methods exactly mimicing those in the paper.

My results are as follows:

- **Cora (*transductive*):** 86% accuracy in ~600 epochs;
- **CiteSeer (*transductive*):** 75% accuracy in ~300 epochs;
- **PubMed (*transductive*):** [] accuracy in ~[] epochs;
- **PPI (*inductive*):** 0.96 micro-F1 score in ~600 epochs.

## Directory structure

The models and training are implemented in a Python package contained in `src/oxgat`. Within this:

- the `components` module defines individual GAT layers and attention heads;
- the `models` module defines trainable models according to the original paper;
- the `utils` module defines additional utilities.

Example notebooks are provided in the `notebooks` directory.

## Package installation

Notebooks in this directory depend on the Python package `oxgat` mentioned above. This should be installed using
```
pip install -e damon/src
```
in the repository root; all dependencies will be automatically added.

## Further comments

My implementations are based on PyTorch Lightning, appropriately abstracting away details of the training process; a class heirarchy is defined in `models` which provides automated training interfaces for each model.

## To do

- Implement sparse tensor operations
- Extend in various directions
