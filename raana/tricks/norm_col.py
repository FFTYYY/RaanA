import torch as tc
from .base import Trick

class TrickNormCol(Trick):

    def __init__(self, rate = 0.003):
        super().__init__()  

        self.rate = rate
        self.d_out = None
    
    def holdout(self, W: tc.Tensor):
        '''W: (d_in, d_out)'''

        d_in, d_out = W.size()
        self.d_out = d_out
        norm = W.norm(dim = 0) # (d_out)

        num_holdout = int(d_out * self.rate)
        _, idx = norm.sort(descending = True)
        idx = idx[:num_holdout]
        
        holdout = (W + 0.)[:,idx].clone().detach()

        W[:,idx] *= 0

        return W, [idx, holdout]
    
    def deholdout(self, X: tc.Tensor, heldout: tc.Tensor):
        '''
            X: (n, d_in)
            heldout = (d_holdedout, d_out)
        '''

        idx , holdout = heldout
        n,d_in = X.size()
        d_out = self.d_out
        
        extra = X.new_zeros(n, d_out)
        extra[:,idx] = X @ holdout.to(X.device)

        return extra


        
        
        
        


