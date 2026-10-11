#ifndef CNN_XILINX_ACTIVATIONS_H
#define CNN_XILINX_ACTIVATIONS_H

#include <cmath>
#include <cassert>

inline float relu(float a)
{
    return a > 0 ? a : 0;
}

/**
 * En versiones con pragmas (p.e. arquitectura baja latencia) descomentar las líneas de pragmatización PIPELINE
 */
inline void softmax(float* input, size_t size) {

    int i;
    float m, sum, constant;

    m = -INFINITY;
    find_max:
    for (i = 0; i < size; ++i) {
        //#pragma HLS PIPELINE II=1
        if (m < input[i]) {
            m = input[i];
        }
    }

    sum = 0.0;
    exp_sum:
    for (i = 0; i < size; ++i) {
        //#pragma HLS PIPELINE II=4
        sum += exp(input[i] - m);
    }

    constant = m + log(sum);
    normalize:
    for (i = 0; i < size; ++i) {
        //#pragma HLS PIPELINE II=1
        input[i] = exp(input[i] - constant);
    }


}



#endif //CNN_XILINX_ACTIVATIONS_H
