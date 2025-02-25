import tensorflow as tf

from utils import *

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, ksize, out_channels, stride, activation_function=tf.nn.relu, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(out_channels, ksize, strides=stride, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.03))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = activation_function

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.batch_norm(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

# Use this instead of conv_layer()
def conv_layer(input_tensor, ksize, in_channels, out_channels, stride, scope_name, activation_function=tf.nn.relu):
    conv = ConvLayer(ksize, out_channels, stride, activation_function)
    return conv(input_tensor)




def residual_layer(input_image, ksize, in_channels, out_channels, stride, scope_name):
    with tf.name_scope(scope_name):  
        output = conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name+"_conv1")  # ✅ Only get `output`
        output = conv_layer(output, ksize, out_channels, out_channels, stride, scope_name+"_conv2")  # ✅ Only get `output`
        output = tf.add(output, tf.identity(input_image))
        return output  # ✅ Now only returning `output`

def transpose_deconvolution_layer(input_tensor, used_weights, new_shape, stride, scope_name):
    with tf.name_scope(scope_name):  # ✅ Fix variable scope issue
        output = tf.nn.conv2d_transpose(input_tensor, used_weights, output_shape=new_shape, strides=[1, stride, stride, 1], padding='SAME')
        output = tf.nn.relu(output)
        return output

def resize_deconvolution_layer(input_tensor, new_shape, scope_name):
    with tf.name_scope(scope_name):  
        output = tf.image.resize(input_tensor, (new_shape[1], new_shape[2]), method='bilinear')  
        output = conv_layer(output, 3, new_shape[3]*2, new_shape[3], 1, scope_name+"_deconv")  # ✅ Now returns only `output`
        return output

def deconvolution_layer(input_tensor, new_shape, scope_name):
    return resize_deconvolution_layer(input_tensor, new_shape, scope_name)

def output_between_zero_and_one(output):
    output +=1
    return output / 2
