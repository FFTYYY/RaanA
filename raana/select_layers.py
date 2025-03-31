import torch as tc
import torch.nn as nn
import transformers
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from typing import Callable
from scipy.linalg import hadamard

def default_linear_selector(layer: nn.Module):
    """Check if a layer is a linear layer.
    """
    flag_1 = False
    try:
        flag_1 = layer.weight is not None
    except AttributeError:
        pass

    flag_2 = isinstance(layer, nn.Linear)

    return flag_1 & flag_2

def get_layers(
    model       : nn.Module, 
    linear_selector : Callable[[nn.Module], bool],
    prefix      : str                         = "", 
    name2father : None | dict[str, nn.Module] = None , 
    name2layer  : None | dict[str, nn.Module] = None , 
    name2suffix : None | dict[str, str]       = None , 
    layer_names : None | list[str]            = None ,
):
    """
    Recursively traverse the model to get all linear layers.
    
    Args:
        model: The model to traverse.
        linear_selector: Function to determine if a layer is linear.
    """
    name2father = {} if name2father is None else name2father
    name2layer  = {} if name2layer  is None else name2layer
    name2suffix = {} if name2suffix is None else name2suffix
    layer_names = [] if layer_names is None else layer_names

    for name, layer in model.named_children():

        layer_name = f"{prefix + '.' if prefix else ''}{name}"
        layer.__name__     = layer_name  # type: ignore

        # TODO ???
        if layer_name == "lm_head":
            continue

        if linear_selector(layer):
            layer_names.append(layer_name)
            name2layer [layer_name] = layer
            name2father[layer_name] = model
            name2suffix[layer_name] = name

            continue # linear layers must be leaf nodes

        name2father, name2layer, name2suffix, layer_names = get_layers(
            layer           , 
            linear_selector , 
            layer_name      , 
            name2father     , 
            name2layer      , 
            name2suffix     , 
            layer_names     ,
        )
    
    return name2father, name2layer, name2suffix, layer_names