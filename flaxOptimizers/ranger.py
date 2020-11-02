import math
import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _RangerHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    alpha: onp.ndarray
    k: onp.ndarray
    N_sma_threshhold: onp.ndarray
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
                       alpha=0.5, k=6, N_sma_threshhold=5, use_gc=True):
        hyper_params = _RangerHyperParams(learning_rate, beta1, beta2, eps, weight_decay, 
                                          alpha, k, N_sma_threshhold, use_gc)
        super().__init__(hyper_params)
        # TODO RAdam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

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
        k = hyper_params.k
        alpha = hyper_params.alpha
        use_gc = hyper_params.use_gc
        N_sma_threshhold = hyper_params.N_sma_threshhold

        # Applies gradient centralization
        grad = _gradient_centralization(grad, use_gc=use_gc)

        # computes exponential moving averages
        grad_sq = lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        #grad_ema_corr = grad_ema / (1 - beta1 ** step)
        #grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** step)

        # TODO RAdam
        step += 1
        buffered = self.radam_buffer[int(step % 10)]
        if step == buffered[0]:
            N_sma = buffered[1]
            step_size = buffered[2]
        else:
            buffered[0] = step
            beta2_t = beta2 ** step
            N_sma_max = 2 / (1 - beta2) - 1
            N_sma = N_sma_max - 2 * step * beta2_t / (1 - beta2_t)
            buffered[1] = N_sma
            if N_sma > N_sma_threshhold:
                step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** step)
            else:
                step_size = 1.0 / (1 - beta1 ** step)
            buffered[2] = step_size

        # computes delta
        if N_sma > N_sma_threshhold:
            denom = jnp.sqrt(grad_sq_ema) + eps
            delta = grad_ema / denom
        else:
            delta = grad_ema

        # weight decay
        delta += param * weight_decay

        # applies gradient
        new_param = param - delta * step_size * learning_rate

        # integrated look ahead
        slow_buffer = state.slow_buffer
        if step % k == 0:
            slow_buffer = (1.0 - alpha)*slow_buffer + alpha*new_param
            # copy interpolated weights to RAdam param tensor
            new_param = slow_buffer

        new_state = _RangerParamState(grad_ema, grad_sq_ema, slow_buffer)
        return new_param, new_state


def _gradient_centralization(x, use_gc=True):
    """concept taken from https://github.com/Yonghongwei/Gradient-Centralization"""
    # TODO we might do the use_gc test in the main loop
    if use_gc and len(list(x.size())) > 1:
        x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True)) # TODO
    return x
