import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _MadgradHyperParams:
    learning_rate: onp.ndarray
    beta: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    use_adamWStyle_weightDecay: onp.bool

@struct.dataclass
class _MadgradParamState:
    initial_param: onp.ndarray
    grad_sum: onp.ndarray
    grad_sum_sq: onp.ndarray

class Madgrad(OptimizerDef):
    """
    MADGRAD: Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization.
    https://github.com/facebookresearch/madgrad

    Differences with reference implementation:
    - this version uses a AdamW style weight decay by default
    - the learning rate has been remapped such that it acts linearly

    Notes:
    - Madgrad with use_adamWStyle_weightDecay=False usually requires less weight decay than other optimizer (often 0).
    - The optimal learning rate will usually not be the one used by SGD or Adam.
      Some good values include :
      - 0.005 for 152 layer PreActResNet on CIFAR10,
      - 0.01 for ResNet-50 on ImageNet,
      - 0.085 for IWSLT14 using transformer_iwslt_de_en,
      - 0.03 for RoBERTa training on BookWiki using BERT_BASE.
    """

    def __init__(self, learning_rate=5e-2, beta=0.9, eps=1e-6, weight_decay=0.0, use_adamWStyle_weightDecay=True):
        hyper_params = _MadgradHyperParams(learning_rate, beta, eps, weight_decay, use_adamWStyle_weightDecay)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _MadgradParamState(param, jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        beta = hyper_params.beta
        weight_decay = hyper_params.weight_decay
        learning_rate = hyper_params.learning_rate
        eps = hyper_params.eps

        # weight decay
        if not hyper_params.use_adamWStyle_weightDecay:
            grad += weight_decay * param

        # gradient accumulation
        # power(3/2) added such that learning_rate is the actual step size rather than lr / cbrt(lr)
        weighted_lr = jnp.power(learning_rate, 3./2.) * jnp.sqrt(step + 1)
        grad_sum = state.grad_sum + weighted_lr * grad
        grad_sum_sq = state.grad_sum_sq + weighted_lr * lax.square(grad)

        # parameter update
        new_param = state.initial_param - grad_sum / (jnp.cbrt(grad_sum_sq) + eps)
        new_param = beta*param + (1. - beta)*new_param # momentum

        # AdamW-style weight decay
        if hyper_params.use_adamWStyle_weightDecay:
            new_param -= (1. - beta) * learning_rate * weight_decay * param

        new_state = _MadgradParamState(state.initial_param, grad_sum, grad_sum_sq)
        return new_param, new_state
