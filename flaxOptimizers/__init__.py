# official implementations, provided for centralization sake
from flax.optim import Adafactor, Adagrad, Adam, LAMB, LARS, Momentum, RMSProp, GradientDescent as SGD, WeightNorm

# optimizers
from .adamhd import AdamHD
from .adamp import AdamP
from .laprop import LaProp, LaPropCentered
from .madgrad import Madgrad
from .madgradSimplified import MadgradSimplified
from .radam import RAdam
from .radamSimplified import RAdamSimplified
from .ranger import Ranger
from .sadam import Sadam

# work in progress
# from .adarem import AdaRem

# my experiments
from .rlaprop import RLaProp

__all__ = ['Adafactor', 'Adagrad', 'Adam', 'LAMB', 'LARS', 'Momentum', 'RMSProp', 'SGD', 'WeightNorm',
           'AdamHD', 'AdamP', 'LaProp', 'LaPropCentered', 'Madgrad', 'MadgradSimplified', 'RAdam', 'RAdamSimplified', 'Ranger', 'Sadam',
           'RLaProp',]
