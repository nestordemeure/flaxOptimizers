# Flax Optimizers

A collection of optimizers for Flax.
The repository is open to pull requests.

## Installation

You can install this librarie with:

```
pip install git+https://github.com/nestordemeure/flaxOptimizers.git
```

## Optimizers

Classical optimizers, inherited from the official Flax implementation:

- [Adafactor](https://arxiv.org/abs/1804.04235) A memory efficient optimizer, has been used for large-scale training of attention-based models.
- [Adagrad](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) Introduces a denominator to SGD so that each parameter has its own learning rate.
- [Adam](https://arxiv.org/abs/1412.6980) The most common stochastic optimizer nowadays.
- [LAMB](https://arxiv.org/abs/1904.00962) Improvement on LARS to makes it efficient across task types.
- [LARS](https://arxiv.org/abs/1708.03888) An optimizer designed for large batch.
- [Momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) SGD with momentum, optionally Nesterov momentum.
- [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) Developped to solve Adagrad's diminushing learning rate problem. 
- [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) The simplest stochastic gradient descent optimizer possible.

More arcane first-order optimizers:

- [AdamHD](https://arxiv.org/abs/1703.04782) Uses hypergradient descent to tune its own learning rate. Good at the begining of the training but tends to underperform at the end.
- [AdamP](https://arxiv.org/abs/2006.08217v2) Corrects premature step-size decay for scale-invariant weights. Useful when a model uses some form of Batch normalization.
- [LapProp](https://arxiv.org/abs/2002.04839) Applies exponential smoothing to update rather than gradient.
- [MADGRAD](https://arxiv.org/abs/2101.11075) Modernisation of the Adagrad family of optimizers, very competitive with Adam.
- [RAdam](https://arxiv.org/abs/1908.03265) Uses a rectified variance estimation to compute the learning rate. Makes training smoother, especially in the first iterations.
- [RAdamSimplified](https://arxiv.org/abs/1910.04209) Warmup strategy proposed to reproduce RAdam's result with a much decreased code complexity.
- [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) Combines look-ahead, RAdam and gradient centralization to try and maximize performances. Designed with picture classification problems in mind.
- [Sadam](https://arxiv.org/abs/1908.00700) Introduces an alternative to the epsilon parameter.

<!--
work in progress:
- [AdaRem](https://arxiv.org/abs/2010.11041v1) Reduce oscilations in update vector.
-->

Optimizer wrappers:

- [WeightNorm](https://arxiv.org/abs/1602.07868) Alternative to BatchNormalization, does the weight normalization inside the optimizer which makes it compatible with more models and faster (*official Flax implementation*)

## Other references

- [AdahessianJax](https://github.com/nestordemeure/AdaHessianJax) contains my implementation of the Adahessian second order optimizer in Flax.
- [Flax.optim](https://github.com/google/flax/tree/master/flax/optim) contains a number of optimizer that currently do not appear in the official documentation. They are all included accesible from this librarie.
