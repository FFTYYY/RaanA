import torch as tc
import torch.nn.functional as F
import math 
from typing import Callable

from .base import RandomRotation, default_hadamard

def sign(x: tc.Tensor):
    return 2 * (x >= 0).float() - 1

def floor_power2(x: int):
    return 2 ** int(math.floor(math.log2(x)))

class PiecewiseHadamard(RandomRotation):
    def __init__(self, hadamard: Callable[[tc.Tensor], tc.Tensor] = default_hadamard):
        super().__init__(hadamard)

        self.rad_1 = None
        self.rad_2 = None
    
    def preprocess(self, X: tc.Tensor):
        device = X.device

        if self.rad_1 is None:
            self.rad_1 = sign(tc.randn(X.size(-1), device = device)) # random +/- 1
            self.rad_2 = sign(tc.randn(X.size(-1), device = device)) # random +/- 1

        return X

    def to(self, device: tc.device) -> "PiecewiseHadamard":
        if self.rad_1 is not None:
            self.rad_1 = self.rad_1.to(device)
            self.rad_2 = self.rad_2.to(device)
        return self
    
    def apply(self, X: tc.Tensor):
        '''X: (n, d)'''

        assert X.size(-1) == self.rad_1.numel()
        hadamard = self.hadamard


        d_in  = int( X.size(-1) )
        downd = int( floor_power2(d_in) )

        X = X * self.rad_1.view(1, -1).to(device = X.device,dtype = X.dtype)
        X[:, : downd] = hadamard(X[:, : downd]).to(X.dtype)

        X = X * self.rad_2.view(1, -1).to(device = X.device,dtype = X.dtype)
        X[:, -downd:] = hadamard(X[:, -downd:]).to(X.dtype)
        
        return X
    
