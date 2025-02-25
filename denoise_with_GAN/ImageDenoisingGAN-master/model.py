import numpy as np
import tensorflow as tf
from tf_slim import layers

from utils import *
from conv_helper import *

@tf.function
def generator(input_tensor):
    """Defines the Generator Network for 768x576 input"""
    
    batch_size = tf.shape(input_tensor)[0]  

    conv1 = ConvLayer(9, 32, 1)(input_tensor)
    conv2 = ConvLayer(3, 64, 1)(conv1)
    conv3 = ConvLayer(3, 128, 1)(conv2)

    res1 = residual_layer(conv3, 3, 128, 128, 1, "g_res1")  
    res2 = residual_layer(res1, 3, 128, 128, 1, "g_res2")
    res3 = residual_layer(res2, 3, 128, 128, 1, "g_res3")

    deconv1 = deconvolution_layer(res3, [batch_size, 384, 288, 64], 'g_deconv1')
    deconv2 = deconvolution_layer(deconv1, [batch_size, 768, 576, 32], "g_deconv2")

    deconv2 = deconv2 + conv1  # Skip connection

    conv4 = conv_layer(deconv2, 9, 32, 3, 1, "g_conv5", activation_function=tf.nn.tanh)
    conv4 = conv4 + input_tensor
    output = output_between_zero_and_one(conv4)

    return output


@tf.function
def discriminator(input_tensor):
    """Defines the Discriminator Network for 768x576 input"""
    
    conv1 = conv_layer(input_tensor, 4, 3, 48, 2, "d_conv1")
    conv2 = conv_layer(conv1, 4, 48, 96, 2, "d_conv2")
    conv3 = conv_layer(conv2, 4, 96, 192, 2, "d_conv3")
    conv4 = conv_layer(conv3, 4, 192, 384, 1, "d_conv4")
    conv5 = conv_layer(conv4, 4, 384, 1, 1, "d_conv5", activation_function=tf.nn.sigmoid)

    return conv5
