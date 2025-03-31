#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <vector>
#include <math.h>

namespace py = pybind11;
using std::vector;
using std::tuple;

double cost(double B){
    if(B >= 1){
        return 1. / pow(2., B);
    }
    return 1. / (2. * sqrt(B));
}

/**
 * sizes : [num_layers]
 * coeffs: [num_layers]
 * nob   : [num_layers]
 * lowbit: if true, use < 1 bit
 * 
 * return: status, results
 * status: 0 for success, -1 for size mismatch, -2 for no solution, -3 for wrong solution
 */
tuple<int, pybind11::array_t<float>> dp(
    int total_r, 
    int num_layers, 
    py::array_t<int> _sizes, 
    py::array_t<float> _coeffs, 
    py::array_t<double> _B_options
) {
    auto _nob = py::array_t<float>(num_layers);
    float * nob = static_cast<float*>(_nob.request().ptr);

    py::buffer_info sizes_buf  = _sizes .request();
    py::buffer_info coeffs_buf = _coeffs.request();
    py::buffer_info B_options_buf = _B_options.request();
    pybind11::ssize_t B_size = B_options_buf.size;

    int     * sizes     = static_cast<int  *>(sizes_buf.ptr);
    float   * coeffs    = static_cast<float*>(coeffs_buf.ptr);
    double  * B_options = static_cast<double*>(B_options_buf.ptr);
    if(sizes_buf.ndim != 1 || coeffs_buf.ndim != 1)
        return std::make_tuple(-1, _nob);
    if(sizes_buf.size != num_layers || coeffs_buf.size != num_layers)
        return std::make_tuple(-1, _nob);

    vector<vector<double>>                f(num_layers, vector<double>        (total_r + 1, -1)           );
    vector<vector<vector<double>>> solution(num_layers, vector<vector<double>>(total_r + 1, vector<double>()));

    for(pybind11::ssize_t B_idx = 0; B_idx < B_size; B_idx ++){
        double B = B_options[B_idx];
        int    extra_r    = int(double(sizes[0]) * B + 0.5);
        double extra_cost = coeffs[0] * cost(B);

        f       [0][extra_r] = extra_cost;
        solution[0][extra_r] = {B};
    }

    for(int layer_idx = 1;layer_idx < num_layers; layer_idx++){
        for(pybind11::ssize_t B_idx = 0; B_idx < B_size; B_idx ++){
            double B = B_options[B_idx];
            int    extra_r    = int(double(sizes[layer_idx]) * B + 0.5);
            double extra_cost = coeffs[layer_idx] * cost(B);

            for(int prev_r = 1;prev_r <= total_r - extra_r; prev_r++){
                if(f[layer_idx - 1][prev_r] < 0)
                    continue;
                if(prev_r + extra_r > total_r) 
                    continue;
                
                double now_cost = f[layer_idx - 1][prev_r] + extra_cost;
                if(f[layer_idx][prev_r + extra_r] < 0 || f[layer_idx][prev_r + extra_r] > now_cost){
                    f       [layer_idx][prev_r + extra_r] = now_cost;
                    solution[layer_idx][prev_r + extra_r] = solution[layer_idx - 1][prev_r];
                    solution[layer_idx][prev_r + extra_r].push_back(B);
                }
            }
        }
    }

    int best_r = -1;
    vector<double> & final_f = f[num_layers - 1];
    for(int i = 0;i <= total_r;i ++){
        if(final_f[i] < 0) 
            continue;
        if(best_r < 0 || final_f[i] < final_f[best_r]){
            best_r = i;
        }
    }

    if(best_r < 0)
        return std::make_tuple(-2, _nob);
    vector<double> & best_sol = solution[num_layers - 1][best_r];
    if(int( best_sol.size() ) != num_layers)
        return std::make_tuple(-3, _nob);
    
    for (int i = 0;i < num_layers;i ++){
        nob[i] = float( best_sol[i] );
    }
    return std::make_tuple(0, _nob);
}

PYBIND11_MODULE(dp, m) {
    m.doc() = "dp determine nob."; 
    m.def("dp", &dp, "dp determine nob.");
}
