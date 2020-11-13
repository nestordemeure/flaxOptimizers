import numpy as onp
import jax.numpy as jnp
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _AdaRemHyperParams:
    learning_rate: onp.ndarray
    beta: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    lr_scaling: onp.ndarray

@struct.dataclass
class _AdaRemParamState:
    grad_ema: onp.ndarray

class AdaRem(OptimizerDef):
    """
    AdaRem
    https://arxiv.org/abs/2010.11041v1
    """

    def __init__(self, learning_rate=0.4, beta=0.9, eps=1e-8, weight_decay=0.0, lr_scaling=0.999):
        """1e-4 is a good baseline for a non-zero weight decay"""
        # TODO proper name for lr_scaling ?
        hyper_params = _AdaRemHyperParams(learning_rate, beta, eps, weight_decay, lr_scaling)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _AdaRemParamState(jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        beta = hyper_params.beta
        weight_decay = hyper_params.weight_decay
        eps = hyper_params.eps
        learning_rate = hyper_params.learning_rate
        lr_scaling = hyper_params.lr_scaling
        grad_ema = state.grad_ema

        # per parameter learning rate
        # TODO does b have a proper name ?
        max_ema_amplitude = jnp.max(jnp.abs(grad_ema))
        b = jnp.sign(grad) * grad_ema / (max_ema_amplitude + eps)
        adjustement_coeffiscient = 1. + (lr_scaling**step) * b

        # update parameters (including weight decay)
        update = adjustement_coeffiscient * grad + weight_decay * param
        new_param = param - learning_rate * update

        # exponential moving average (no need for bias correction)
        grad_ema = beta * grad_ema + (1. - beta) * grad
        new_state = _AdaRemParamState(grad_ema)
        return new_param, new_state
