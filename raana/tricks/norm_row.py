import torch as tc
from .base import Trick

def idx2mask(idx: tc.Tensor, size: int):
    mask = tc.zeros(size).bool()
    mask[idx] = 1
    return mask

class TrickNormRow(Trick):

    def __init__(self, rate = 0.003):
        super().__init__()  

        self.rate = rate
        self.d_in = None
    
    def holdout(self, W: tc.Tensor):
        '''W: (d_in, d_out)'''

        d_in, d_out = W.size()
        self.d_in = d_in
        norm = W.norm(dim = 1) # (d_in)

        num_holdout = int(d_in * self.rate)
        _, idx = norm.sort(descending = True)
        idx = idx[:num_holdout]
        
        holdout = (W[idx] + 0.).clone().detach()

        W[idx2mask(idx, d_in)] = 0

        return W, [idx, holdout]
    
    def deholdout(self, X: tc.Tensor, heldout: tc.Tensor):
        '''
            X: (n, d_in)
            heldout = (d_holdedout, d_out)
        '''

        idx , holdout = heldout

        extra = X[:,idx] @ holdout.to(X.device)

        return extra


        
        
        
        


