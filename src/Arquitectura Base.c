#include "params.h"
#include "activations.h"
#include "top_cnn.h"

void fc(float out[], float in[], float weight[FC_WEIGHTS_H][FC_WEIGHTS_W], float bias[FC_BIAS_SIZE]) {
    float read;
    float output[FC_ACT_SIZE] = {0};

    
    bucle_fc1:
    for (int j = 0; j < FC_WEIGHTS_H; j++) {
        read = in[j];
        bucle_fc11:
        for (int i = 0; i < FC_WEIGHTS_W; i++) {            
            output[i] += weight[j][i] * read;
        }
    }

    bucle_fc2:
    for (int i = 0; i < FC_WEIGHTS_W; i++){
        out[i] = output[i] + bias[i];
    }

    softmax(out, FC_ACT_SIZE);
}

void pool(float out[], float in[]) {
    int BUFFER_SIZE = (P_SIZE * P_CHANNELS);

    int i, j, k, l, m;
    float read;
    float pool_buff[BUFFER_SIZE][1];
    int z = 0;
    int out_index = 0;

    bucle_pool1:
    for (i = 0; i < P_SIZE; i++)
        bucle_pool11:
        for (l = 0; l < P_KERNEL_SIZE; l++) {
            bucle_pool12:
            for (j = 0; j < P_SIZE; j++)
                bucle_pool13:
                for (m = 0; m < P_KERNEL_SIZE; m++)            
                    bucle_pool14:
                    for (k = 0; k < P_CHANNELS; k++) {       
                        read = in[z];
                        z++;
                        if (l == 0 && m == 0)
                            pool_buff[j * P_CHANNELS + k][0] = read;
                        else
                            pool_buff[j * P_CHANNELS + k][0] =
                                pool_buff[j * P_CHANNELS + k][0] > read ? pool_buff[j * P_CHANNELS + k][0] : read;
                        if (l == (P_KERNEL_SIZE - 1) && m == (P_KERNEL_SIZE - 1)) {
                            out[out_index] = pool_buff[j * P_CHANNELS + k][0];
                            out_index++;
                        }
                    }
            bucle_pool15:
            for (int skip = P_SIZE * P_STRIDE; skip < A_SIZE; skip++)
                    bucle_pool16:
                for (int channel = 0; channel < P_CHANNELS; channel++) {
                    z++;
                }
        }
}

void conv(float out[], const float in[], const float weight[CONV_KERNEL_SIZE][CONV_KERNEL_SIZE][CONV_CHANNELS][CONV_FILTERS], float bias[CONV_BIAS_SIZE]) {
    int BUFFER_SIZE = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS;
    int i, j, k, filter;
    float sum;
    int row_offset, col_offset, channel_offset;

    float conv_buff[IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS];

    bucle_conv1:
    for (i = 0; i < BUFFER_SIZE; i++) {
        conv_buff[i] = in[i];
    }

    int out_index = 0;
    bucle_conv2:
    for (i = 0; i < (IMAGE_SIZE - CONV_KERNEL_SIZE + 1); i += CONV_STRIDE) {
        bucle_conv21:
        for (j = 0; j < (IMAGE_SIZE - CONV_KERNEL_SIZE + 1); j += CONV_STRIDE) {
            bucle_conv22:
            for (filter = 0; filter < CONV_FILTERS; filter++) {
                sum = 0;
                bucle_conv23:
                for (row_offset = 0; row_offset < CONV_KERNEL_SIZE; row_offset++)          
                    bucle_conv24:
                    for (col_offset = 0; col_offset < CONV_KERNEL_SIZE; col_offset++)
                        bucle_conv25:
                        for (channel_offset = 0; channel_offset < CONV_CHANNELS; channel_offset++) 
                            sum += conv_buff[row_offset * IMAGE_SIZE * IMAGE_CHANNELS +
                                             col_offset * IMAGE_CHANNELS +
                                             channel_offset +
                                             i * (IMAGE_SIZE) + j] *
                                   weight[row_offset][col_offset][channel_offset][filter];
                out[out_index] = relu(sum + bias[filter]);
                out_index++;
            }
        }
    }
}

void pool(float out[], float in[]);
void fc(float out[], float in[], float weight[FC_WEIGHTS_H][FC_WEIGHTS_W], float bias[FC_BIAS_SIZE]);

void top_cnn(
    float image[28 * 28],
    float fc_out[10],
    float conv_weights[3][3][1][32],
    float conv_bias[32],
    float fc_weights[5408][10],
    float fc_bias[10]
) {
    float conv_out[A_SIZE * A_SIZE * A_CHANNELS];
    float pool_out[P_SIZE * P_SIZE * P_CHANNELS];

    conv(conv_out, image, conv_weights, conv_bias);
    pool(pool_out, conv_out);
    fc(fc_out, pool_out, fc_weights, fc_bias);
}
