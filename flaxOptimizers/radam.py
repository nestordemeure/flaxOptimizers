import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _RAdamHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    n_sma_threshhold: onp.ndarray

@struct.dataclass
class _RAdamParamState:
    grad_ema: onp.ndarray
    grad_sq_ema: onp.ndarray


class RAdam(OptimizerDef):
    """
    RAdam optimizer
    Uses a rectified variance estimation to compute the learning rate.
    https://arxiv.org/abs/1908.03265
    """

    def __init__(self, learning_rate=None, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 n_sma_threshhold=4):
        """n_sma_threshhold is the threshold above which the full formula will be used, we recomend setting it to 4 or 5"""
        hyper_params = _RAdamHyperParams(learning_rate, beta1, beta2, eps, weight_decay, n_sma_threshhold)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _RAdamParamState(jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay
        n_sma_threshhold = hyper_params.n_sma_threshhold
        grad_sq = lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        t = step + 1.
        grad_ema_corr = grad_ema / (1 - beta1 ** t)

        # RAdam update
        n_sma_inf = 2. / (1 - beta2) - 1.
        n_sma_t = n_sma_inf - (2. * t * beta2 ** t) / (1. - beta2 ** t)
        if n_sma_t > n_sma_threshhold: # low variance
            # step size computation
            step_size_num = (n_sma_t - 4.) * (n_sma_t - 2.) * n_sma_inf
            step_size_denum = (n_sma_inf - 4.) * (n_sma_inf - 2.) * n_sma_t
            step_size = jnp.sqrt(step_size_num / step_size_denum)
            # update tensor computation
            grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)
            denom = jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps
            update = step_size * grad_ema_corr / denom
        else: # hight variance
            update = grad_ema_corr

        new_param = param - hyper_params.learning_rate * update
        new_param -= hyper_params.learning_rate * weight_decay * param
        new_state = _RAdamParamState(grad_ema, grad_sq_ema)
        return new_param, new_state
