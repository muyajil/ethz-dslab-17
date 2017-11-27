import tensorflow as tf
from tensorflow.python.framework import ops


def batch_norm(x,
               eps=1e-5,
               momentum=0.9,
               name="batch_norm"):
    return tf.contrib.layers.batch_norm(x,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=eps,
                                        scale=True,
                                        scope=name)


def conv2d(input_,
           out_channels,
           kernel_height=5,
           kernel_width=5,
           stride_height=2,
           stride_width=2,
           stddev=0.02,
           name="conv2d"):
    """ Convolutional Layer with default stride 2x2.
    """
    with tf.variable_scope(name):
        in_channels = input_.get_shape()[-1]
        w = tf.get_variable('w',
                            [kernel_height, kernel_width, in_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_,
                            w,
                            strides=[1, stride_height, stride_width, 1],
                            padding='SAME')
        biases = tf.get_variable('biases',
                                 [out_channels],
                                 initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())


def deconv2d(input_,
             output_shape,
             kernel_height=5,
             kernel_width=5,
             stride_height=2,
             stride_width=2,
             stddev=0.02,
             name="deconv2d",
             with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [kernel_height, kernel_width, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_,
                                        w,
                                        output_shape=output_shape,
                                        strides=[1, stride_height, stride_width, 1])
        biases = tf.get_variable('biases',
                                 [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
            
def binarization(x):
    g = tf.get_default_graph()
    with ops.name_scope("Binarization") as name:
        with g.gradient_override_map({"Ceil": "Identity", "Sub": "CustomGrad", "Div": "CustomGrad", "Add": "CustomGrad", "Mul": "CustomGrad"}):
            scaled_x = (x + 1) / 2
            binary_x = tf.ceil(scaled_x - tf.random_uniform(tf.shape(x)), name=name)
            return (binary_x * 2) - 1
       
@ops.RegisterGradient("CustomGrad")
def customGrad(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]


def psnr(original, reconstructed, peak=1.0, scope=None):
    mse = tf.reduce_mean(tf.squared_difference(original, reconstructed))
    rmse = tf.sqrt(mse)
    return tf.scalar_mul(tf.constant(20.0), log10(tf.divide(tf.constant(peak), rmse)))


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return tf.divide(numerator, denominator)
