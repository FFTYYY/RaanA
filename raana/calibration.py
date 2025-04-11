import torch as tc
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer, LlamaForCausalLM
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm 
from typing import Callable, Any

from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from .task_adaptor import TaskAdaptor, LMAdaptor

zeroshot_calibration = LMAdaptor([
    ("The curious fox leaped over the quiet stream, "
        + "its reflection rippling in the golden afternoon light. ") * 100,
])

class Calibrator:

    def __init__(self, 
        model           : nn.Module,
        calibrate_data  : TaskAdaptor,

        layer_names     : list[str],
        name2layer      : dict[str, nn.Module],

    ):
        self.model          = model
        self.layer_names    = layer_names
        self.name2layer     = name2layer
        self.calibrate_data = calibrate_data

        self.initialize()


    def initialize(self):
        self.jacob_norms: dict[str, list[float]] = {}
        self.x_sizes    : dict[str, int]         = {}
        self.w_sizes    : dict[str, int]         = {} # real size
        self.norm_infos : dict[str, list[float]] = {}
        self.loss_vals  : list[float]            = []

    def jacobian_hook(self, layer, grad_input, grad_output):
        name = layer.__name__

        grad_o = grad_output[0]

        if name not in self.jacob_norms:
            self.jacob_norms[name] = []

        self.jacob_norms[name].append( float( grad_o.norm() ) )
        self.x_sizes    [name] = int( grad_o.view(-1).size(0) )

    def norm_hook(self, layer, input, output):
        name = layer.__name__

        X: tc.Tensor = input[0].view(-1, input[0].size(-1))
        W: tc.Tensor = layer.weight
        n, d_in = X.size()
        if W.size(0) != d_in:
            W = W.t()
        assert W.size(0) == d_in

        self.w_sizes[name] = int(W.numel())

        '''The following two lines are equivalent to the following two lines:
                N_1 = (X.norm(dim = 1) ** 2).mean()
                N_2 = (W.norm(dim = 0) ** 2).sum() / W.size(0)
            We use this writing style to prevent from possible float point overflow.
        '''
        N_1 = ((X.to(tc.float64).norm(dim = 1) / (n ** 0.5)) ** 2).sum()
        N_2 = ((W.to(tc.float64).norm(dim = 0) / (W.size(0)**0.5)) ** 2).sum()

        norm_info =  (N_1 ** 0.5) * (N_2 ** 0.5)

        if name not in self.norm_infos:
            self.norm_infos[name] = []
        self.norm_infos[name].append(float(norm_info))


    def register_hooks(self) -> dict[str, tuple[RemovableHandle, RemovableHandle]]:
        name2hooks = {}
        for name in self.layer_names:
            layer = self.name2layer[name]
            handle_1 = layer.register_full_backward_hook(self.jacobian_hook)
            handle_2 = layer.register_forward_hook      (self.norm_hook    )
            name2hooks[name] = [handle_1, handle_2]
        return name2hooks

    def remove_hooks(self, name2hooks: dict[str, tuple[RemovableHandle, RemovableHandle]]):
        for name in self.layer_names:
            handle_1, handle_2  = name2hooks[name]
            handle_1.remove()
            handle_2.remove()

    def backward(self, model: nn.Module, datapoint: str):

        loss = self.calibrate_data.get_loss(model, datapoint)
        model.zero_grad()
        loss.backward()
        model.zero_grad()
        return [ float(loss) ]


    def run(self) -> tuple[dict[str, float], dict[str, float]]:

        self.initialize()

        name2hooks = self.register_hooks()
        loss_vals = []
        for datapoint in self.calibrate_data.data:
            loss_vals.append(self.backward(self.model, datapoint))
        self.loss_vals = loss_vals
        self.remove_hooks(name2hooks)

        layer_names = self.layer_names
        jacob_norms = self.jacob_norms
        x_sizes     = self.x_sizes
        w_sizes     = self.w_sizes
        norm_infos  = self.norm_infos

        sensitivities = {
            name: np.mean(jacob_norms[name]) / ( x_sizes[name] ** 0.5) 
            for name in layer_names
        }
        norm_infos = {
            name: np.mean(norm_infos[name])
            for name in layer_names
        }

        return sensitivities, norm_infos, w_sizes
