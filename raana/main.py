import torch as tc
from torch import nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import Callable

from .select_layers import get_layers, default_linear_selector
from .rotations import RandomRotation, default_rotation
from .tricks import Trick
from .quantized_linear import QuantizedLinear
from .quantized_linear import default_weightbias_extractor, default_matmul
from .calibration import Calibrator
from .dp import determine_nob
from .task_adaptor.base import TaskAdaptor

class Quantizer:
    def __init__(self, 
            model: PreTrainedModel, 
            linear_selector : Callable[[nn.Module], bool] = default_linear_selector,
        ):
        self.model = model

        name2father, name2layer, name2suffix, layer_names = get_layers(model, linear_selector)

        self.name2father = name2father
        self.name2layer  = name2layer
        self.name2suffix = name2suffix
        self.layer_names = layer_names

    def initialize(self, 
        rotation_maker    : Callable[[], RandomRotation] = default_rotation,
        trick_makers: list[Callable[[], Trick]] = [],
        weightbias_extractor: Callable[
            [nn.Module], tuple[tc.Tensor, tc.Tensor | None]
        ] = default_weightbias_extractor,
        matmul: Callable[[tc.Tensor, tc.Tensor, tc.Tensor, int], tc.Tensor] = default_matmul,
    ):
        pbar = tqdm(self.layer_names)
        for layer_idx, layer_name in enumerate(pbar):
            pbar.set_description(f"now initializing {layer_name}")

            layer  = self.name2layer [layer_name]
            father = self.name2father[layer_name]
            suffix = self.name2suffix[layer_name]

            quantized_layer = QuantizedLinear(
                rotation_maker  = rotation_maker,
                trick_makers    = trick_makers,
                weightbias_extractor = weightbias_extractor,
                matmul          = matmul,
            )
            quantized_layer = quantized_layer.initialize_from_layer(layer)
            quantized_layer.__name__ = layer_name

            father.__setattr__(suffix, quantized_layer)
            self.name2layer[layer_name] = quantized_layer
        
        return self.model

    def apply_quantization(self, nob: dict[str, int]):

        pbar = tqdm(self.layer_names, desc = "quantizing")
        for layer_idx, layer_name in enumerate(pbar):
            pbar.set_description(f"now quantizing {layer_name}")

            layer  = self.name2layer [layer_name]
            B      = nob[layer_name]

            assert isinstance(layer , QuantizedLinear)
            layer = layer.set_B(B)
            layer = layer.quantize()
            
            for i in range(tc.cuda.device_count()):
                with tc.cuda.device(i):
                    tc.cuda.empty_cache()
            
        return self.model

    def determine_nob(self, 
            B           : float, 
            b_candidates : list[float], 
            calibrate_data: TaskAdaptor, 
        ):

        for bc in b_candidates:
            if bc <= 0 or (bc > 1 and abs(bc - round(bc)) > 1e-4):
                raise ValueError(f"b_candidate = {bc} is not a valid value.")

        model       = self.model
        layer_names = self.layer_names
        name2layer  = self.name2layer

        calibrator = Calibrator(model, calibrate_data, layer_names, name2layer)
        sensitivities, norm_infos, w_sizes = calibrator.run()

        
        nob = determine_nob(
            B, 
            layer_names, 
            w_sizes, 
            sensitivities, 
            norm_infos,
            b_candidates,
        )

        return nob, calibrator.loss_vals
