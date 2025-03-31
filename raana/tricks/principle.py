import torch as tc
from .base import Trick

class TrickPCA(Trick):

    def __init__(self, rate = 0.003):
        super().__init__()  

        self.rate = rate
    
    def holdout(self, W: tc.Tensor):
        '''W: (d_in, d_out)'''

        dtype = W.dtype

        U,S,V = tc.svd(W.to(tc.float32)) # W = U @ S.diag() @ V.t()
        # W = sum_i U[:,i].view(-1,1) * V[:,i].view(1,-1) * S[i] 

        num_holdout = int(S.numel() * self.rate)
        
        Us = U[:,:num_holdout].clone().detach()
        Vs = V[:,:num_holdout].clone().detach()
        Ss = S[:num_holdout].clone().detach()

        rem_U = U[:,num_holdout:]
        rem_V = V[:,num_holdout:]
        rem_S = S[num_holdout:]

        new_W = rem_U @ rem_S.diag() @ rem_V.t()

        return new_W.to(dtype) [Us.to(dtype), Vs.to(dtype), Ss.to(dtype)]

    
    def deholdout(self, X: tc.Tensor, heldout: tc.Tensor):
        '''
            X: (n, d_out)
            heldout = [Us, Vs, Ss]
            Us: (d_in, num_holdout)
            Ss: (num_holdout,)
            Vs: (d_out, num_holdout)
        '''
        Us, Vs, Ss = heldout
        return X @ (Us @ Ss.diag() @ Vs.t()).to(X.dtype).to(X.device)
    
