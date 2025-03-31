import torch as tc
from .base import Trick

class TrickCentralize(Trick):

    def __init__(self):
        super().__init__()  

    
    def holdout(self, W: tc.Tensor):
        '''W: (d_in, d_out)'''
        
        mu_W = W.mean(dim = 1).view(-1)    # (d_in)
        W = W - mu_W.view(-1, 1)           # (d_in, d_out)
        mu_W = mu_W.clone().detach()


        return W, mu_W
    
    def deholdout(self, X: tc.Tensor, mu_W: tc.Tensor):
        '''
            X: (n, d_in)
            mu_W: (d_in)
        '''
        d_in = mu_W.size(0)
        extra =  X @ mu_W.view(d_in, 1).to(X.device)

        return extra
    