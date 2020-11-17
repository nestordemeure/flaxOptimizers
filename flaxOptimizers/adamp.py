import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _AdamPHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    delta: onp.ndarray
    wd_ratio: onp.ndarray

@struct.dataclass
class _AdamPParamState:
    grad_ema: onp.ndarray
    grad_sq_ema: onp.ndarray

class AdamP(OptimizerDef):
    """
    AdamP optimizer
    https://github.com/clovaai/AdamP/blob/master/adamp/adamp.py
    """

    def __init__(self, learning_rate=None, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 delta=0.1, wd_ratio=0.1):
        """
        `delta`: threhold that determines whether a set of parameters is scale invariant or not
        `wd_ratio`: relative weight decay applied on scale-invariant parameters compared to that applied on scale-variant parameters
        the default for both values should be fairly reliable
        """
        hyper_params = _AdamPHyperParams(learning_rate, beta1, beta2, eps, weight_decay,
                                         delta, wd_ratio)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _AdamPParamState(jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay
        delta = hyper_params.delta
        wd_ratio = hyper_params.wd_ratio
        eps = hyper_params.eps
        grad_sq = lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        t = step + 1.
        grad_ema_corr = grad_ema / (1 - beta1 ** t)
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

        # compute update
        denom = jnp.sqrt(grad_sq_ema_corr) + eps
        update = grad_ema_corr / denom

        # Projection
        if len(param.shape) > 1:
            update, wd_ratio = _project_scale_invariant(update, param, grad, delta, wd_ratio, eps)
        else:
            wd_ratio = 1.

        # weight decay
        new_param = param * (1. - hyper_params.learning_rate * weight_decay * wd_ratio)

        # update step
        new_param -= hyper_params.learning_rate * update
        new_state = _AdamPParamState(grad_ema, grad_sq_ema)
        return new_param, new_state

def _channel_view(x):
    channel_shape = (x.shape[0], -1)
    return x.reshape(channel_shape)

def _layer_view(x):
    layer_shape = (1, -1)
    return x.reshape(layer_shape)

def _projection(update, param, param_view, eps):
    """projection onto the tangent space of param"""
    reexpanded_shape = [-1] + [1] * (param.ndim - 1)
    view_norm = jnp.linalg.norm(param_view, axis=1).reshape(reexpanded_shape)
    param_normalized = param / (view_norm + eps)
    return update - jnp.sum(param_normalized * update, axis=1, keepdims=True) * param_normalized

def _cosine_similarity(x, y, eps):
    """cosine similarity between two tensors"""
    dotxy = jnp.sum(x * y, axis=1)
    normx = jnp.linalg.norm(x, axis=1)
    normy = jnp.linalg.norm(y, axis=1)
    return jnp.abs(dotxy) / jnp.maximum(eps, normx * normy)

def _is_scale_invariant(param, grad, delta, eps):
    """test to determine if the tensor is scale invariant"""
    return jnp.max(_cosine_similarity(param, grad, eps)) < delta / jnp.sqrt(param.shape[1])

def _make_view_conditionalupdate(update, param, grad, delta, view_func, eps):
    """
    produces a test and an update vector, if the test is true then one should use the update vector
    works with a view of the parameters
    """
    param_view = view_func(param)
    grad_view = view_func(grad)
    cond = _is_scale_invariant(param_view, grad_view, delta, eps)
    update = _projection(update, param, param_view, eps)
    return (cond, update)

def _project_scale_invariant(update, param, grad, delta, wd_ratio, eps):
    """projects only if the parameters are detected to be scale invariant in channel or layer view"""
    cond_channel, update_channel = _make_view_conditionalupdate(update, param, grad, delta, _channel_view, eps)
    cond_layer, update_layer = _make_view_conditionalupdate(update, param, grad, delta, _layer_view, eps)
    # use one of the suggested projection if a scale invariant tensor has been detected
    update = jnp.where(cond_channel, update_channel, jnp.where(cond_layer, update_layer, update))
    # use a wd_ratio if update has been modified
    wd_ratio = jnp.where(cond_channel | cond_layer, wd_ratio, 1.)
    return (update, wd_ratio)
