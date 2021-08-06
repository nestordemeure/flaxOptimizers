from functools import partial
import numpy as onp
import jax.numpy as jnp
from jax import lax
from jax.nn import softplus
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _Ranger21HyperParams:
    learning_rate: onp.ndarray
    beta0: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    use_softplus: onp.ndarray
    beta_softplus: onp.ndarray
    eps_clipping: onp.ndarray
    threshold_clipping: onp.ndarray
    weight_decay: onp.ndarray
    beta_lookahead: onp.ndarray
    lookahead_every_nth_iter: onp.ndarray
    nb_iterations: onp.ndarray
    nb_warmup_iterations: onp.ndarray
    nb_warmdown_iterations: onp.ndarray
    centralize_gradients: onp.ndarray
    normalize_gradients: onp.ndarray

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
    
    This implementation follows the [code](https://github.com/lessw2020/Ranger21)
    which has been updated since the publication of the [paper](https://arxiv.org/abs/2106.13731)
    
    In particular it uses gradient normalization (unpublished) instead of gradient centralization
    and the idea of using a softplus instead of an epsilon to deal with zero denominators (https://arxiv.org/abs/1908.00700)
    """

    def __init__(self, nb_iterations,
                 learning_rate=1e-3, 
                 beta0=0.9, beta1=0.9, beta2=0.999, 
                 eps=1e-8, use_softplus=False, beta_softplus=50,
                 eps_clipping=1e-3, threshold_clipping=1e-2,
                 weight_decay=1e-4,
                 beta_lookahead=0.5, lookahead_every_nth_iter=5,
                 nb_warmup_iterations=None, nb_warmdown_iterations=None,
                 centralize_gradients=True, normalize_gradients=True):
        """
        beta0: Manages the amplitude of the noise introduced by positive negative momentum.
               While 0.9 is a recommended default value, you can use -0.5 to minimize the noise.
        weight_decay: the optimizer is much more sensitve to weight decay than AdamW and thus, it should be limited to lower values such as 1e-4
        """
        nb_warmup_iterations = (0.22 * nb_iterations) if nb_warmup_iterations is None else nb_warmup_iterations
        nb_warmdown_iterations = (0.28 * nb_iterations) if nb_warmdown_iterations is None else nb_warmdown_iterations
        hyper_params = _Ranger21HyperParams(learning_rate, beta0, beta1, beta2, 
                                            eps, use_softplus, beta_softplus,
                                            eps_clipping, threshold_clipping,
                                            weight_decay, beta_lookahead, lookahead_every_nth_iter,
                                            nb_iterations, nb_warmup_iterations, nb_warmdown_iterations,
                                            centralize_gradients, normalize_gradients)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _Ranger21ParamState(jnp.zeros_like(param), jnp.zeros_like(param), jnp.zeros_like(param), jnp.zeros_like(param), param)

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        # gets the optimizer parameters
        t = step + 1.
        beta0 = hyper_params.beta0
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        beta1_squared = beta1 * beta1
        non_zero = partial(_non_zero, eps=hyper_params.eps, use_softplus=hyper_params.use_softplus, beta_softplus=hyper_params.beta_softplus)

        # prepares gradient
        grad = _gradient_clipping(grad, param, non_zero, hyper_params.eps_clipping, hyper_params.threshold_clipping)
        grad = _gradient_normalization(grad, non_zero, hyper_params.centralize_gradients, hyper_params.normalize_gradients)

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
        pnm_noise_amplitude = jnp.hypot(1. + beta0, beta0) # takes positive negative momentum into account
        denom = pnm_noise_amplitude * jnp.sqrt(grad_sq_ema_corr)
        update = grad_ema_corr / non_zero(denom)

        # weight decay
        # combining norm-loss and stable weight decay
        euclidian_norm = _axis_aware_euclidian_norm(param) # for norm-loss regularization
        effective_stepsize_inv = jnp.sqrt( jnp.mean(grad_sq_ema_corr) ) # for stable weight decay
        scaled_weight_decay = hyper_params.weight_decay * (euclidian_norm - 1.) / non_zero(euclidian_norm*effective_stepsize_inv)
        update += scaled_weight_decay * param

        # applies update
        scheduled_learning_rate = _learning_rate_scheduler(hyper_params.learning_rate, t, hyper_params.nb_iterations, 
                                                           hyper_params.nb_warmup_iterations, hyper_params.nb_warmdown_iterations, 
                                                           beta2)
        new_param = param - update * scheduled_learning_rate

        # look-ahead
        (new_param, lookahead_ema) = _lookahead(new_param, state.lookahead_ema, t, hyper_params.beta_lookahead, hyper_params.lookahead_every_nth_iter)

        new_state = _Ranger21ParamState(state.grad_ema, grad_ema, grad_sq_ema, grad_sq_ema_max, lookahead_ema)
        return new_param, new_state

def _non_zero(x, eps=1e-8, use_softplus=False, beta_softplus=50, threshold_softplus=20):
    """insures that a value is non-zero either by applying a softplus or adding an epsilon to it"""
    def smooth_softplus(x, beta, threshold=threshold_softplus): 
        """
        sofplus function but with additional control over the smoothness via the beta parameter
        threshold is there for numerical stability
        """
        return jnp.where(x > threshold_softplus, x, softplus(beta * x) / beta)
    return smooth_softplus(x, beta_softplus) if use_softplus else (x + eps)

def _axis_aware_euclidian_norm(param):
    """euclidian norm with special cases to deal with various layer shapes"""
    if param.ndim <= 1:
        # fully flattens the norm
        return jnp.linalg.norm(param, ord=2, keepdims=False)
    else:
        # dimensions along which to compute the norm, special case for linear layers
        axis = 1 if param.ndim <= 3 else tuple(range(1, param.ndim)) # 1 ... ndim-1
        return jnp.linalg.norm(param, ord=2, axis=axis, keepdims=True)

def _gradient_clipping(grad, param, non_zero, eps=1e-3, threshold=1e-2):
    """
    variant of gradient clipping that uses a dynamic threshold
    `eps` is there to avoid freezing zero-parameters
    `non_zero` is a function that takes an input and insures that it will not be zero or negative
    """
    norm_grad = non_zero(_axis_aware_euclidian_norm(grad))
    norm_param = lax.max(_axis_aware_euclidian_norm(param), eps)
    dynamic_threshold = threshold * (norm_param / norm_grad)
    return jnp.where(dynamic_threshold < 1., grad * dynamic_threshold, grad)

def _gradient_normalization(grad, non_zero, centralize_gradients=True, normalize_gradients=True):
    """
    substract the mean from the gradient and divide it by its standard deviation
    `non_zero` is a function that takes an input and insures that it will not be zero or negative
    """
    can_centralize = centralize_gradients and (grad.ndim > 1)
    can_normalize = normalize_gradients and (grad.size > 2)
    if can_centralize or can_normalize:
        # takes into account the fact that the gradient might be 1D
        keepdims = (grad.ndim > 1)
        axis = tuple(range(1, grad.ndim)) if keepdims else None
        # substract the mean from the gradient
        grad_mean = grad.mean(axis=axis, keepdims=keepdims)
        grad -= grad_mean
        if can_normalize:
            # divide the centralized gradient by its standard deviation
            grad_std = grad.std(axis=axis, keepdims=keepdims)
            grad /= non_zero(grad_std) # we divide *after* subtracting the mean
            # add the mean back to the gradient if we don't want to centralize it
            if not can_centralize: grad += grad_mean
    return grad

def _learning_rate_scheduler(max_learning_rate, 
                             iteration, nb_iterations, nb_warmup_iterations, nb_warmdown_iterations, 
                             beta2):
    """combines explore-exploit scheduling with a linear warmup"""
    warmup_scaling = lax.max(0.5 * iteration * (1. - beta2), iteration / nb_warmup_iterations)
    warmdown_scaling = (nb_iterations - iteration) / nb_warmdown_iterations
    scaling = lax.min(1., lax.min(warmup_scaling, warmdown_scaling))
    return scaling * max_learning_rate

def _lookahead(param, lookahead_ema, step, beta_lookahead=0.5, lookahead_every_nth_iter=4):
    """lookahead at the param level instead of group level"""
    condition = step % lookahead_every_nth_iter < 0.5 # == 0. but inexact to deal with roundoffs
    lookahead_ema = jnp.where(condition, beta_lookahead*lookahead_ema + (1. - beta_lookahead)*param, lookahead_ema)
    param = jnp.where(condition, lookahead_ema, param)
    return (param, lookahead_ema)
