from jax.nn import sigmoid

__all__ = ['is_greater', 'is_mod', 'gpu_cond']

def is_greater(x, threshold, amplifier=1e10):
    """1 if x > threshold, 0 otherwise"""
    return sigmoid(amplifier * (x - threshold))

def is_mod(x, mod_n, amplifier=1e10):
    """1 if x % n == 0, 0 otherwise"""
    return is_greater(0.5, x % mod_n, amplifier)

def gpu_cond(cond, y_true, y_false):
    """
    a GPU friendly alternative to lax.cond
    if cond=1, returns y_true
    if cond=0, returns y_false
    use when computing both branches is cheap
    """
    return y_true*cond + y_false*(1. - cond)
