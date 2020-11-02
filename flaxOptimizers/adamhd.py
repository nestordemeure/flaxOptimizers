import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _AdamHDHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    hypergrad_lr: onp.ndarray

@struct.dataclass
class _AdamHDParamState:
    grad_ema: onp.ndarray
    grad_sq_ema: onp.ndarray
    learning_rate: onp.float32

class AdamHD(OptimizerDef):
    """
    AdamHD optimizer: AdamW + hypergradient descent
    https://github.com/gbaydin/hypergradient-descent
    """

    def __init__(self, learning_rate=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, hypergrad_lr=1e-8):
        hyper_params = _AdamHDHyperParams(learning_rate, beta1, beta2, eps, weight_decay, hypergrad_lr)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _AdamHDParamState(jnp.zeros_like(param), jnp.zeros_like(param), self.hyper_params.learning_rate)

    def apply_param_gradient(self, iteration, hyper_params, param, state, grad):
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay

        # exponential averaging
        grad_sq = lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        t = iteration + 1.
        grad_ema_corr = grad_ema / (1 - beta1 ** t)
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

        # new step (both gradient and weight decay) to be applied
        step = grad_ema_corr / ( jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps ) + weight_decay * param

        # hypergradient computation and descent
        hypergrad = jnp.sum(grad * step)
        learning_rate = state.learning_rate + hypergrad * hyper_params.hypergrad_lr

        new_param = param - learning_rate * step
        new_state = _AdamHDParamState(grad_ema, grad_sq_ema, learning_rate)
        return new_param, new_state
