import math
import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

from .utilities import gpu_cond

@struct.dataclass
class _RangerHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    beta_lookahead: onp.ndarray
    lookahead_every_nth_iter: onp.ndarray
    n_sma_threshhold: onp.ndarray
    use_gc: onp.ndarray

@struct.dataclass
class _RangerParamState:
    grad_ema: onp.ndarray
    grad_sq_ema: onp.ndarray
    slow_buffer: onp.ndarray

class Ranger(OptimizerDef):
    """
    Ranger optimizer.
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger2020.py
    """

    def __init__(self, learning_rate=1e-3, beta1=0.95, beta2=0.999, eps=1e-5, weight_decay=0.0,
                       beta_lookahead=0.5, lookahead_every_nth_iter=6, n_sma_threshhold=5, use_gc=True):
        hyper_params = _RangerHyperParams(learning_rate, beta1, beta2, eps, weight_decay,
                                          beta_lookahead, lookahead_every_nth_iter, n_sma_threshhold, use_gc)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _RangerParamState(jnp.zeros_like(param), jnp.zeros_like(param), param)

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        # gets the optimizer parameters
        learning_rate = hyper_params.learning_rate
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        eps = hyper_params.eps
        weight_decay = hyper_params.weight_decay
        beta_lookahead = hyper_params.beta_lookahead
        lookahead_every_nth_iter = hyper_params.lookahead_every_nth_iter
        n_sma_threshhold = hyper_params.n_sma_threshhold

        # Applies gradient centralization
        grad = _gradient_centralization(grad, use_gc=hyper_params.use_gc)

        # computes exponential moving averages
        grad_sq = lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        t = step + 1.
        grad_ema_corr = grad_ema / (1 - beta1 ** t)
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

        # RAdam update
        n_sma_inf = 2. / (1 - beta2) - 1.
        n_sma_t = n_sma_inf - (2. * t * beta2 ** t) / (1. - beta2 ** t)
        # step size computation
        step_size_num = (n_sma_t - 4.) * (n_sma_t - 2.) * n_sma_inf
        step_size_denum = (n_sma_inf - 4.) * (n_sma_inf - 2.) * n_sma_t
        # abs added to deal with negative values in first iterations (happens when the test will ignore step_size anyway)
        step_size = jnp.sqrt( jnp.abs(step_size_num / step_size_denum) )
        denom = jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps
        # update tensor computation
        update = gpu_cond(n_sma_t, n_sma_threshhold, # n_sma_t > n_sma_threshhold
                          step_size * grad_ema_corr / denom, # true
                          grad_ema_corr) # false

        # weight decay
        update += param * weight_decay

        # applies update
        new_param = param - update * learning_rate

        # integrated look ahead
        (new_param, slow_buffer) = _lookahead(new_param, state.slow_buffer, step, beta_lookahead, lookahead_every_nth_iter)

        new_state = _RangerParamState(grad_ema, grad_sq_ema, slow_buffer)
        return new_param, new_state

def _gradient_centralization(grad, use_gc=True):
    """concept taken from https://github.com/Yonghongwei/Gradient-Centralization"""
    if use_gc and grad.ndim > 1:
        averaging_dimensions = tuple(range(1, grad.ndim)) # all except 0 axis
        grad -= grad.mean(axis=averaging_dimensions, keepdims=True)
    return grad

def _lookahead(param, slow_buffer, step, beta_lookahead=0.5, lookahead_every_nth_iter=4):
    """lookahead at the param level instead of group level"""
    if step % lookahead_every_nth_iter == 0:
        slow_buffer = beta_lookahead*slow_buffer + (1.0 - beta_lookahead)*param
        param = slow_buffer
    return (param, slow_buffer)
