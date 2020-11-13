import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct, linen

@struct.dataclass
class _SadamHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    weight_decay: onp.ndarray
    smooth: onp.ndarray

@struct.dataclass
class _SadamParamState:
    grad_ema: onp.ndarray
    grad_sq_ema: onp.ndarray

class Sadam(OptimizerDef):
    """
    Sadam optimizer
    https://github.com/neilliang90/Sadam/blob/master/optimizer/Sadam.py
    """

    def __init__(self, learning_rate=None, beta1=0.9, beta2=0.999, eps=None, weight_decay=0.0, smooth=50):
        # use value at origin to convert epsilon into smooth parameter
        if not (eps is None): smooth = onp.log(2.) / eps
        hyper_params = _SadamHyperParams(learning_rate, beta1, beta2, weight_decay, smooth)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _SadamParamState(jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay
        smooth = hyper_params.smooth
        grad_sq = lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        t = step + 1.
        grad_ema_corr = grad_ema / (1 - beta1 ** t)
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

        # Sadam alternative to eps
        denom = smooth_softplus(jnp.sqrt(grad_sq_ema_corr), smooth)

        new_param = param - hyper_params.learning_rate * grad_ema_corr / denom
        new_param -= hyper_params.learning_rate * weight_decay * param
        new_state = _SadamParamState(grad_ema, grad_sq_ema)
        return new_param, new_state

def smooth_softplus(x, smooth=50):
    """becomes relu when smooth tends to infinity"""
    return linen.softplus(x * smooth) / smooth
