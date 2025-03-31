import torch as tc
import torch.nn as nn
from typing import Callable, NewType

from .select_layers import default_linear_selector
from .task_adaptor.base import TaskAdaptor
from .task_adaptor.lm import LMAdaptor
from .rotations import RandomRotation, default_rotation
from .tricks import Trick
from .quantized_linear import default_weightbias_extractor, default_matmul
from .main import Quantizer
from .calibration import zeroshot_calibration # export for convenience
from .tricks import trick_centralize, trick_norm_row, trick_norm_col, trick_pca

def quantize(
    model           : nn.Module   , 
    b_candidates    : list[float] , 
    calibrate_data  : TaskAdaptor , 
    avg_bits        : float       ,
    linear_selector : Callable[[nn.Module], bool]  = default_linear_selector,
    rotation_maker  : Callable[[], RandomRotation] = default_rotation,
    trick_makers    : list[Callable[[], Trick]]    = [trick_centralize, trick_norm_col],
    weightbias_extractor : Callable[ [nn.Module], tuple[tc.Tensor, tc.Tensor | None]] 
                                                   = default_weightbias_extractor,
    matmul          : Callable[[tc.Tensor, tc.Tensor, tc.Tensor, int], tc.Tensor]        
                                                   = default_matmul,
):
    trick_makers = [*trick_makers] # copy the list to avoid side effects

    quantizer   = Quantizer(model,linear_selector)
    model       = quantizer.initialize(rotation_maker, trick_makers, weightbias_extractor, matmul)
    nob, losses = quantizer.determine_nob(avg_bits, b_candidates, calibrate_data)
    model       = quantizer.apply_quantization(nob)
    return {
        "model" : model,
        "bits"  : nob,
        "losses": losses,
    }
