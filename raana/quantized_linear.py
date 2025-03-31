import torch as tc
import torch.nn as nn
from typing     import Callable
from .rotations  import RandomRotation, default_rotation
from .tricks     import trick_centralize, trick_pca, trick_norm_row, trick_norm_col
from .tricks     import Trick

from .rabitq import quantize

def default_weightbias_extractor(layer: nn.Module) -> tuple[tc.Tensor, tc.Tensor | None]:
    if isinstance(layer, nn.Linear):
        return (
            layer.weight.data.t(), 
            layer.bias.data.view(-1) if layer.bias is not None else None
        )
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}.")
    return "fuck"

def default_matmul(X: tc.Tensor, qW: tc.Tensor, rescale: tc.Tensor, B: int):
    dtype = X.dtype
    X       = X.to(tc.float32)
    rescale = rescale.to(tc.float32).view(1, -1)
    q_bias  = (float(2 ** B - 1) / 2. * X.sum(dim = -1)).view(-1, 1)
    Z = (X @ qW.to(tc.float32)) * rescale
    Z = Z - q_bias * rescale
    return Z.to(dtype)

class QuantizedLinear(nn.Module):
    def __init__(self, 
        rotation_maker  : Callable[[], RandomRotation],
        trick_makers    : list[Callable[[], Trick]],
        weightbias_extractor: Callable[
            [nn.Module], tuple[tc.Tensor, tc.Tensor | None]
        ] = default_weightbias_extractor,
        matmul          : Callable[[tc.Tensor, tc.Tensor, tc.Tensor, int], tc.Tensor] = default_matmul,
    ):
        '''
            weight: (d_in, d_out)
            bias  : (d_out, )
        '''
        super().__init__()

        self.rotation       = rotation_maker()
        self.tricks         = [_() for _ in trick_makers]
        self.wb_extractor   = weightbias_extractor
        self.matmul         = matmul

        self.B          = -1
        self.weight     = None # (d_in, d_out)
        self.bias       = None # (d_out, )

        self.qW         = None
        self.rescale    = None
        self.JLdim      = -1  
    
    @tc.no_grad()
    def _initialize(self, weight: tc.Tensor, bias: tc.Tensor | None):
        '''
            weight: (d_in, d_out)
            bias  : (d_out, ) or None
        '''
        for trick in self.tricks:
            weight = trick.pre_quantize(weight)
        self.weight = weight.data.detach() 
        
        if bias is not None:
            self.bias = bias.data.detach() 

        return self

    def initialize_from_layer(self, layer: nn.Module):
        weight, bias = self.wb_extractor(layer)
        device = weight.device

        self._initialize(weight, bias)
        self.to(device)
        return self
    
    @tc.no_grad()
    def quantize(self): 
        assert self.weight is not None

        rotator = self.rotation
        W = self.weight 
        B = self.B

        assert B > 0

        PW = rotator.apply(rotator.preprocess(W.t())).t() # (d_in, d_out)
        d_in, d_out  = PW.size()
        
        if B >= 0.99:
            B = int(B + 0.1)
            rescale, qW = quantize(PW.data , B = B)  # (d_out,) , (d_in, d_out)

            self.rescale = rescale
            self.qW      = qW
        else:
            d_in = PW.size(0)

            self.JLdim = int(d_in * B + 0.5)
            JLrescale = d_in / self.JLdim
            PW = PW[:self.JLdim] 

            rescale, qW = quantize(PW.data , B = 1)

            self.rescale = rescale * JLrescale
            self.qW      = qW

        self.weight = None # free memory
        return self
    
    def set_B(self, B: float):
        self.B = B
        return self 

    def to(self, device):
        if self.weight is not None:
            self.weight     = self.weight.to(device)
        if self.bias is not None:
            self.bias       = self.bias.to(device)
        if self.qW is not None:
            self.qW   = self.qW.to(device)
        if self.rescale is not None:
            self.rescale   = self.rescale.to(device)
        self.rotation = self.rotation.to(device)
        return self
    
    def is_quantized(self):
        return self.qW is not None

    def is_initialized(self):
        return self.weight is not None

    @property
    def device(self):
        if self.weight is not None:
            return self.weight.device
        return self.qW.device

    def forward_prequantize(self, X: tc.Tensor):
        assert self.is_initialized()

        W = self.weight
        _, d_out = W.size()

        target_size = X.size()[:-1] + (d_out, ) # (bs, n, d_out)
        X = X.view(-1,X.size(-1)) # (bs*n, d_in)
        
        new_X = X @ W # (n, d_out)
        
        for trick in self.tricks:
            new_X = new_X + trick.post_quantize(X) # use original X to centralize
            
        if self.bias is not None:
            new_X = new_X + self.bias.view(1,-1)

        new_X = new_X.view(target_size)
        return new_X
        
    @tc.no_grad()
    def forward_quantize(self, X: tc.Tensor):
        '''
            X: (n,d_in)
        '''

        assert self.is_quantized()

        qW      = self.qW  # (d_in, d_out)
        
        B       = self.B       # (d_out, )
        _, d_out = qW.size()

        rotator     = self.rotation
        matmul      = self.matmul

        target_size = X.size()[:-1] + (d_out, ) # (bs, n, d_out)
        X = X.view(-1,X.size(-1)) # (bs*n, d_in)

        # ----- preprocess X -----

        rot_X   = rotator.apply(rotator.preprocess(X))  # (bs*n, d_in)
        if self.JLdim > 0:
            rot_X = rot_X[:, : self.JLdim] 

        # ----- quantize matmul -----
        
        new_X = matmul(rot_X, qW, self.rescale, max(B, 1)) # (bs*n, d_out)

        # ----- postprocess -----
        for trick in self.tricks:
            new_X = new_X + trick.post_quantize(X) # use original X to centralize

        if self.bias is not None:
            new_X = new_X + self.bias.view(1,-1)

        ret = new_X.view(target_size)
        return ret
    

    def forward(self, X: tc.Tensor):

        if not self.is_quantized():
            return self.forward_prequantize(X)
        return self.forward_quantize(X)