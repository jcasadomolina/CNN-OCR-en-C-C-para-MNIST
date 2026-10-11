#ifndef TOP_CNN_H
#define TOP_CNN_H

void top_cnn( float image[28 * 28], 
			float fc_out[10],
			float conv_weights[3][3][1][32],
			float conv_bias[32],
			float fc_weights[5408][10],
			float fc_bias[10]);
   
#endif // TOP_CNN_H
