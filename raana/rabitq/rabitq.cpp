#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdint.h>
#include <omp.h>
#include <queue>
#include <tuple>
#include <vector>
#include <cstdio>
#include <cmath>  // For some simple mathematical operations



void compute_extended_code(float * o, int D, int B, uint32_t* code, float * rescale){
    
    constexpr double eps = 1e-5;
    constexpr std::array<float, 9> kTightStart = {
        0,
        0.15,
        0.20,
        0.52,
        0.59,
        0.71,
        0.75,
        0.77,
        0.81,
    };
    // Find the maximum coordinate the vector.
    float max_o = -1;
    
    // printf("Unit Vector: \n");
    // for(int i=0;i<D;i++)printf("%.2f ", o[i]);
    // printf("\n");

    for(int i=0;i<D;i++)if(o[i] > max_o) max_o = o[i];
    double t_end = (double)(((1 << B) - 1) + 10) / max_o;
    double t_start = t_end * kTightStart[B];

    int *cur_code = new int [D];

    double sqr_denominator = D * 0.25;
    double numerator = 0;

    for(int i=0;i<D;i++){
        cur_code[i] = int((double)t_start * o[i] + eps);
        sqr_denominator += (int)cur_code[i] * cur_code[i] + cur_code[i];
        numerator += (cur_code[i] + 0.5) * o[i];
    }    

    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>>> next_t;

    for(int i=0;i<D;i++){
        next_t.emplace(std::make_pair((double)(cur_code[i] + 1) / o[i], i));
    }

    double max_ip = 0;
    double t = 0;

    int cnt = 0;

    while(next_t.empty() == false){
        double cur_t = next_t.top().first;
        int update_id = next_t.top().second;
        ++cnt;
        next_t.pop();

        cur_code[update_id] ++;
        int update_code = cur_code[update_id];
        sqr_denominator += 2 * update_code;
        numerator += o[update_id];
        
        double cur_ip = numerator / std::sqrt(sqr_denominator);
        if(cur_ip > max_ip){
            max_ip = cur_ip;
            t = cur_t;
        }

        if (update_code < (1 << B)-1){
            double t_next = (double)(update_code + 1) / o[update_id];
            if(t_next < t_end) next_t.emplace(std::make_pair(t_next, update_id));
        }
        
    }

    sqr_denominator = D * 0.25;
    numerator = 0;
    for(int i=0;i<D;i++){
        code[i] = int((double)t * o[i] + eps);
        if (code[i] >= (1 << B))code[i]=(1<<B)-1;
        sqr_denominator += (int)code[i] * code[i] + code[i];
        numerator += (code[i] + 0.5) * o[i];
    }

    // printf("Quantization Code: \n");
    // for(int i=0;i<D;i++)printf("%d ", code[i]);
    // printf("\n");    

    // float norm = std::sqrt(sqr_denominator);
    // float ip = numerator / norm;

    // printf("Inner Product: %lf\n", ip);

    *rescale = 1 / numerator;

    delete [] cur_code;
}


void compute_code(float * data, int D, int B, uint32_t * code, float * rescale){

    memset(code, 0, D * sizeof(uint32_t));

    float norm = 0;
    for(int i=0;i<D;i++){
        norm += data[i] * data[i];
    }
    norm = std::sqrt(norm);

    float * o = new float [D];

    if(norm > 1e-5){
        for(int i=0;i<D;i++){
            code[i] |= ((data[i] > 0) << (B - 1));
            o[i] = (data[i] > 0 ? data[i] : -data[i]);
            o[i] = o[i] / norm;
        }
    }
    else {
        memset(o, 0, D * sizeof(int));
        o[0] = 1;
    }

    // not sure whether there would be problems if B == 1, so I leave it as a special case in the code
    // YYY: 你妈
    if (B > 1){ 
        uint32_t * excode = new uint32_t [D];
        uint32_t mask = (1 << (B - 1)) - 1;

        compute_extended_code(o, D, B - 1, excode, rescale);

        for(int i=0;i<D;i++){
            if(code[i] > 0){
                code[i] |= excode[i];
            }
            else {
                code[i] |= ((~excode[i]) & mask);
            }
        }
        
        *rescale = *rescale * norm;

        delete [] excode;
    }
    else {
        float tmp_ip = 0;
        for(int i=0;i<D;i++)tmp_ip += o[i] / std::sqrt(D);
        *rescale = 2 / std::sqrt(D) * norm / tmp_ip;
    }
    delete [] o;
}

// Function that accepts N*D matrix and returns a tuple: (1D float array, N*D uint32 matrix)
std::tuple<pybind11::array_t<float>, pybind11::array_t<uint32_t>> quantize(const pybind11::array_t<float>& input_array, const size_t B) {
    // Request buffer info to access input array details
    pybind11::buffer_info buf = input_array.request();

    // Ensure the input array is 2D
    if (buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }

    // Get the number of rows (N) and columns (D)
    size_t N = buf.shape[0];
    size_t D = buf.shape[1];

    // Pointer to input data
    float* input_ptr = static_cast<float*>(buf.ptr);

    // Create an output 1D float array of size N (for the coordinates)
    pybind11::array_t<float> coord_array({pybind11::ssize_t(N)});
    float* coord_ptr = static_cast<float*>(coord_array.request().ptr);

    // Create an output 2D uint32_t array of size N*D
    pybind11::array_t<uint32_t> uint32_matrix({N, D});
    uint32_t* uint32_ptr = static_cast<uint32_t*>(uint32_matrix.request().ptr);

    #pragma omp parallel for
    for(size_t i=0;i<N;i++){
        compute_code(input_ptr + i * D, D, B, uint32_ptr + i * D, coord_ptr + i);
    }

    // Return the tuple (1D float array, 2D uint32_t array)
    return std::make_tuple(coord_array, uint32_matrix);
}


PYBIND11_MODULE(rabitq, m) {
    m.def("quantize", &quantize, "A function that processes an N*D matrix and returns (1D float array, N*D uint32 matrix)");
}

