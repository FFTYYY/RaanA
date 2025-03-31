from .dp import dp as _dp # type: ignore
from math import gcd
import torch as tc

def dp(
    total_r     : int, 
    num_layers  : int, 
    sizes       : list[int], 
    coeffs      : list[float], 
    B_options   : list[float]
) -> tuple[int, list[float]]:
    return _dp(total_r, num_layers, sizes, coeffs, B_options)


def determine_nob(
    average_bits    : float, 
    layer_names     : list[str], 
    w_sizes         : dict[str, int], 
    sensitivities   : dict[str, float], 
    norm_infos      : dict[str, float],

    b_candidates    : list[float],

) -> dict[str, float]:
    """
    Determine the number of bits for each layer.
    """
    num_layers = len(layer_names)

    w_sizes_list = [w_sizes[k] for k in layer_names]
    gcd_sizes = gcd( * w_sizes_list, * w_sizes_list )

    dp_bugget = int( sum([_ // gcd_sizes for _ in w_sizes_list]) * average_bits ) # 
    dp_sizes  = tc.LongTensor ([
        w_sizes[k] // gcd_sizes  for k in layer_names
    ] ).numpy().astype("int32") 
    dp_coeffs = tc.FloatTensor([
        sensitivities[k] * norm_infos[k]  for k in layer_names
    ]).numpy().astype("float32")

    assert len(dp_sizes ) == num_layers, f"dp_sizes len error: { len(dp_sizes )} != {num_layers}"
    assert len(dp_coeffs) == num_layers, f"dp_coeffs len error: {len(dp_coeffs)} != {num_layers}"

    flag, nob_list = dp(dp_bugget, num_layers, dp_sizes, dp_coeffs, b_candidates)

    if flag > 0:
        assert False, f"dp error {flag}"
    
    return {
        layer_names[i]: float(nob_list[i])
        for i in range(num_layers)
    }
    