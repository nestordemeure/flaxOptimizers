from jax import nn

__all__ = ['gpu_cond']

def gpu_cond(x, threshold, y_true, y_false, sigmoid_amplifier=1e5):
    """
    a GPU friendly alternative to lax.cond
    returns y_true if x > threshold and y_false otherwise
    use when computing both branches is cheap and the condition is a scalar comparison
    """
    is_above_threshold = nn.sigmoid(sigmoid_amplifier * (x - threshold)) # 1 if x > threshold, 0 otherwise
    return y_true*is_above_threshold + y_false*(1. - is_above_threshold)
