import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _MadgradSimplifiedHyperParams:
    learning_rate: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray

@struct.dataclass
class _MadgradSimplifiedParamState:
    initial_param: onp.ndarray
    grad_sum: onp.ndarray
    grad_sum_sq: onp.ndarray

class MadgradSimplified(OptimizerDef):
    """
    Trying to simplify the Madgrad optimizer by refactoring it into a simpler form
    fusing beta and the learning rate using rewrites that suppose that the learnign rate is constant

    the resulting code should behave similarly but has one less parameter (beta)
    and a learning rate that behaves much closer to usual learning rates
    """

    def __init__(self, learning_rate=5e-3, eps=1e-6, weight_decay=0.0):
        hyper_params = _MadgradSimplifiedHyperParams(learning_rate, eps, weight_decay)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _MadgradSimplifiedParamState(param, jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        # weight decay
        grad += hyper_params.weight_decay * param

        # gradient accumulation
        weight_step = jnp.sqrt(step + 1)
        grad_sum = state.grad_sum + weight_step * grad
        grad_sum_sq = state.grad_sum_sq + weight_step * lax.square(grad)

        # parameter update
        update = grad_sum / (jnp.cbrt(grad_sum_sq) + hyper_params.eps)
        new_param = state.initial_param - hyper_params.learning_rate * update

        new_state = _MadgradSimplifiedParamState(state.initial_param, grad_sum, grad_sum_sq)
        return new_param, new_state
