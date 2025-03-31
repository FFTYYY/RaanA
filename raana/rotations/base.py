import torch as tc
from scipy.linalg import hadamard
from typing import Callable

_default_hadamard_cache = {}
def default_hadamard(X: tc.Tensor) -> tc.Tensor:
    n = X.size(-1)
    if n not in _default_hadamard_cache:
        _default_hadamard_cache[n] = tc.FloatTensor( hadamard(n) ) * (n ** -0.5)
    H = _default_hadamard_cache[n].to(X.device).to(X.dtype)
    return X @ H

class RandomRotation:
    def __init__(self, hadamard: Callable[[tc.Tensor], tc.Tensor] = default_hadamard):
        self.hadamard = hadamard
        
    def preprocess(self, X: tc.Tensor) -> tc.Tensor:
        raise NotImplementedError
    
    def apply(self, X: tc.Tensor) -> tc.Tensor:
        raise NotImplementedError
    
    def to(self, device: tc.device) -> "RandomRotation":
        raise NotImplementedError
