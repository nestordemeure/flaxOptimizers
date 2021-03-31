import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _RAdamSimplifiedHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    use_exponential_warmup: onp.ndarray

@struct.dataclass
class _RAdamSimplifiedParamState:
    grad_ema: onp.ndarray
    grad_sq_ema: onp.ndarray

class RAdamSimplified(OptimizerDef):
    """
    RAdamSimplified optimizer, reproduces the behaviour of RAdam but is much simpler
    We use the linear warmup by default but also provide the exponential warmup.
    https://arxiv.org/abs/1910.04209
    """

    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, use_exponential_warmup=False):
        """
        Same parameters as Adam.
        Set `use_exponential_warmup` to `True` if you want to use an exponential warmup instead of the linear warmup.
        """
        hyper_params = _RAdamSimplifiedHyperParams(learning_rate, beta1, beta2, eps, weight_decay, use_exponential_warmup)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _RAdamSimplifiedParamState(jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay
        grad_sq = lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        t = step + 1.
        grad_ema_corr = grad_ema / (1 - beta1 ** t)
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

        # learning rate warmup, to deal with unstability during first iterations
        exponential_warmup = 1. - jnp.exp( - (1. - beta2) * t )
        linear_warmup = jnp.minimum(1., 0.5 * (1. - beta2) * t)
        learning_rate = hyper_params.learning_rate * jnp.where(hyper_params.use_exponential_warmup, exponential_warmup, linear_warmup)

        denom = jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps
        new_param = param - learning_rate * grad_ema_corr / denom
        new_param -= learning_rate * weight_decay * param
        new_state = _RAdamSimplifiedParamState(grad_ema, grad_sq_ema)
        return new_param, new_state
