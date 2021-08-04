import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

# TODO gradient standardisation
# TODO softplus
# TODO document inputs
# TODO document fact that this implem follows latest code rather than paper

@struct.dataclass
class _Ranger21HyperParams:
    learning_rate: onp.ndarray
    beta0: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    eps_clipping: onp.ndarray
    weight_decay: onp.ndarray
    beta_lookahead: onp.ndarray
    lookahead_every_nth_iter: onp.ndarray
    nb_iterations: onp.ndarray
    nb_warmup_iterations: onp.ndarray
    nb_warmdown_iterations: onp.ndarray
    use_gc: onp.ndarray

@struct.dataclass
class _Ranger21ParamState:
    grad_previous_ema: onp.ndarray
    grad_ema: onp.ndarray
    grad_sq_ema: onp.ndarray
    grad_sq_ema_max: onp.ndarray
    lookahead_ema: onp.ndarray

class Ranger21(OptimizerDef):
    """
    Ranger21 optimizer.
    https://github.com/lessw2020/Ranger21
    https://arxiv.org/abs/2106.13731
    """

    def __init__(self, nb_iterations,
                 learning_rate=1e-3, 
                 beta0=0.9, beta1=0.9, beta2=0.999, 
                 eps=1e-8, eps_clipping=1e-3, 
                 weight_decay=1e-4,
                 beta_lookahead=0.5, lookahead_every_nth_iter=5,
                 nb_warmup_iterations=None, nb_warmdown_iterations=None,
                 use_gc=True):
        nb_warmup_iterations = (0.22 * nb_iterations) if nb_warmup_iterations is None else nb_warmup_iterations
        nb_warmdown_iterations = (0.28 * nb_iterations) if nb_warmdown_iterations is None else nb_warmdown_iterations
        hyper_params = _Ranger21HyperParams(learning_rate, beta0, beta1, beta2, eps, eps_clipping, weight_decay,
                                            beta_lookahead, lookahead_every_nth_iter,
                                            nb_iterations, nb_warmup_iterations, nb_warmdown_iterations, use_gc)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _Ranger21ParamState(jnp.zeros_like(param), jnp.zeros_like(param), jnp.zeros_like(param), jnp.zeros_like(param), param)

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        # gets the optimizer parameters
        t = step + 1.
        beta0 = hyper_params.beta0
        beta1 = hyper_params.beta1
        beta1_squared = beta1 * beta1
        beta2 = hyper_params.beta2

        # prepares gradient
        grad = _gradient_clipping(grad, eps=hyper_params.eps_clipping)
        grad = _gradient_centralization(grad, use_gc=hyper_params.use_gc)

        # first moment estimation
        # using positive-negative momentum and bias correction
        grad_ema = beta1_squared * state.grad_previous_ema + (1. - beta1_squared) * grad
        grad_ema_corr = ((1. + beta0) * grad_ema - beta0 * state.grad_ema) / (1. - beta1 ** t)

        # second moment estimation
        # using positive-negative momentum and bias correction
        grad_sq = lax.square(grad)
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq
        grad_sq_ema_max = lax.max(state.grad_sq_ema_max, grad_sq_ema)
        grad_sq_ema_corr = grad_sq_ema_max / (1 - beta2 ** t)

        # update vector
        beta_normalizer = jnp.hypot(1. + beta0, beta0) # takes positive negative momentum into account
        denom = jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps
        update = grad_ema_corr / (beta_normalizer * denom)

        # TODO weight decay
        update += param * hyper_params.weight_decay

        # applies update
        scheduled_learning_rate = _learning_rate_scheduler(hyper_params.learning_rate, t, hyper_params.nb_iterations, 
                                                           hyper_params.nb_warmup_iterations, hyper_params.nb_warmdown_iterations, 
                                                           beta2)
        new_param = param - update * scheduled_learning_rate

        # look-ahead
        (new_param, lookahead_ema) = _lookahead(new_param, state.lookahead_ema, t, hyper_params.beta_lookahead, hyper_params.lookahead_every_nth_iter)

        new_state = _Ranger21ParamState(state.grad_ema, grad_ema, grad_sq_ema, grad_sq_ema_max, lookahead_ema)
        return new_param, new_state

def _gradient_clipping(grad, eps=1e-3):
    # TODO implement gradient clipping
    return grad

def _gradient_centralization(grad, use_gc=True):
    """concept taken from https://github.com/Yonghongwei/Gradient-Centralization"""
    if use_gc and grad.ndim > 1:
        averaging_dimensions = tuple(range(1, grad.ndim)) # all except 0 axis
        grad -= grad.mean(axis=averaging_dimensions, keepdims=True)
    return grad

def _learning_rate_scheduler(max_learning_rate, 
                             iteration, nb_iterations, nb_warmup_iterations, nb_warmdown_iterations, 
                             beta2):
    """combines explore-exploit scheduling with a linear warmup"""
    warmup_scaling = jnp.max(0.5 * iteration * (1. - beta2), iteration / nb_warmup_iterations)
    warmdown_scaling = (nb_iterations - iteration) / nb_warmdown_iterations
    scaling = jnp.min(1., warmup_scaling, warmdown_scaling) 
    return scaling * max_learning_rate

def _lookahead(param, lookahead_ema, step, beta_lookahead=0.5, lookahead_every_nth_iter=4):
    """lookahead at the param level instead of group level"""
    condition = step % lookahead_every_nth_iter < 0.5 # == 0. but inexact to deal with roundoffs
    lookahead_ema = jnp.where(condition, beta_lookahead*lookahead_ema + (1. - beta_lookahead)*param, lookahead_ema)
    param = jnp.where(condition, lookahead_ema, param)
    return (param, lookahead_ema)
