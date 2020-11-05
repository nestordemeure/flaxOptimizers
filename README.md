# Flax Optimizers

A collection of optimizers for Flax.
The repository is open to pull requests.

## Installation

You can install this librarie with:

```
pip install git+https://github.com/nestordemeure/flaxOptimizers.git
```

## Optimizers

- [AdamHD](https://arxiv.org/abs/1703.04782) Uses hypergradient descent to fit its own learning rate. Good at the begining of the training but tend to underperform at the end.
- [LapProp](https://arxiv.org/abs/2002.04839) Decouples momentum and adaptivity to deal with larger range of input parameters
- [RAdam](https://arxiv.org/abs/1908.03265) Uses a rectified variance estimation to compute the learning rate.
- [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) Combines look-ahead, RAdam and gradient centralization to try and maximize performances.

## Other references

- [Flax.optim](https://github.com/google/flax/tree/master/flax/optim) contains a number of optimizer that currently do not appear in the documentation.
- [AdahessianJax](https://github.com/nestordemeure/AdaHessianJax) contains my implementation of the Adahessian second order optimizer in Flax.
