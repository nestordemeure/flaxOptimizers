import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _LaPropHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray

@struct.dataclass
class _LaPropParamState:
    update_ema: onp.ndarray
    grad_sq_ema: onp.ndarray
    bias_correction1: float

class LaProp(OptimizerDef):
    """
    LaProp optimizer
    https://github.com/Z-T-WANG/LaProp-Optimizer
    """

    def __init__(self, learning_rate=None, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        hyper_params = _LaPropHyperParams(learning_rate, beta1, beta2, eps, weight_decay)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _LaPropParamState(jnp.zeros_like(param), jnp.zeros_like(param), 0.0)

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        learning_rate = hyper_params.learning_rate
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        eps = hyper_params.eps
        weight_decay = hyper_params.weight_decay

        # exponential moving average for grad²
        grad_sq = lax.square(grad)
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq
        # bias correction
        bias_correction2 = 1 - beta2 ** (step + 1.)
        grad_sq_ema_corr = grad_sq_ema / bias_correction2

        # exponential moving average for update tensor
        update = grad / (jnp.sqrt(grad_sq_ema_corr) + eps)
        update_ema = beta1 * state.update_ema + (1. - beta1) * learning_rate * update
        # bias correction
        bias_correction1 = beta1 * state.bias_correction1 + (1 - beta1) * learning_rate
        update_ema_corr = update_ema / bias_correction1

        new_param = param - learning_rate * update_ema_corr
        new_param -= learning_rate * weight_decay * param
        new_state = _LaPropParamState(update_ema, grad_sq_ema, bias_correction1)
        return new_param, new_state
