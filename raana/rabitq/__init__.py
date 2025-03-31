from .rabitq import quantize as _quantize # type: ignore
import torch as tc
import numpy as np

def quantize(
    W: tc.Tensor, 
    B: int , 
) -> tuple[tc.Tensor, tc.Tensor]:
    device = W.device

    W = W.t().cpu().contiguous()
    rescale , code = _quantize( W , B )
    code = code.astype(np.uint8)

    code    = tc.ByteTensor (code).to(device) # (d_out, d_in)
    rescale = tc.FloatTensor(rescale).to(device).to(W.dtype) # (d_out, )

    return rescale, code.t().contiguous()
