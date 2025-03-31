import torch as tc

class Trick:
    def __init__(self):
        self.holdouted = None
        
    def holdout(self, W: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
        '''W: (d_in, d_out)'''

        raise NotImplementedError

    def deholdout(self, X: tc.Tensor, holdouted: tc.Tensor) -> tc.Tensor:
        '''X: (n, d_out)'''

        raise X @ holdouted

    @tc.no_grad()
    def pre_quantize(self, W: tc.Tensor) ->  tc.Tensor:
        '''W: (d_in, d_out)'''

        new_W, holdouted = self.holdout(W)
        self.holdouted = holdouted

        return new_W

    @tc.no_grad()
    def post_quantize(self, X: tc.Tensor):
        '''X: (n, d_in)'''

        assert self.holdouted is not None

        extra = self.deholdout(X, self.holdouted)
        extra = extra.clone().detach().requires_grad_(False)

        return extra