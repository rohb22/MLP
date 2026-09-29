# MLP

A Multi-Layer Perceptron implemented from scratch, inspired by Andrej Karpathy's `micrograd`.

## What's implemented

The MLP is already implemented. From here, we can build on top of it and experiment with different optimizations and ideas, like vectorization, parallelism, batch processing, multiple activation functions, and other loss functions.

## Key decisions

- I cached the parameters list. This works for now since we're not trying to alter the network architecture. It's a basic optimization that would need to be changed if we later allow the architecture to change. Other than that, this follows `micrograd` closely.
