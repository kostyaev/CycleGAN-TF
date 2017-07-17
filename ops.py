import tensorflow as tf
from tensorflow.contrib import slim


def instance_norm_not_trainable(input_var, epsilon=1e-5, axis=(1,2)):
    mean, var = tf.nn.moments(input_var, axis, keep_dims=True)
    return (input_var - mean) / tf.sqrt(var+epsilon)


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


class BatchNorm(object):
    def __init__(self, fused=True, epsilon=1e-5, momentum=0.9, is_training=True):
        self.epsilon = epsilon
        self.momentum = momentum
        self.fused = fused
        self.is_training = is_training

    def __call__(self,input_var, name):
        with tf.variable_scope(name):
            return tf.contrib.layers.batch_norm(input_var,
                                            data_format='NHWC',
                                            fused=self.fused,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            center=True,
                                            is_training=self.is_training,
                                            scope=name)

class BatchNormOld(object):
    def __init__(self, fused=True, epsilon=1e-5, momentum=0.9, is_training=True):
        self.epsilon = epsilon
        self.momentum = momentum
        self.fused = fused
        self.is_training = is_training

    def __call__(self,input_var, name):
        with tf.variable_scope(name):
            return tf.layers.batch_normalization(input_var,
                                                 epsilon=self.epsilon,
                                                 momentum=self.momentum,
                                                 training=self.is_training,
                                                 name=name
                                                 )


def conv2d(x, n_out, ks, stride=1, padding='SAME', name='conv2d', stddev=0.02, activation=tf.nn.relu, normalization=None, dilation=1):
    with tf.variable_scope(name):
        x = slim.conv2d(x, n_out, ks, stride, padding=padding, activation_fn = None,
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev), rate=dilation)
        if normalization:
            x = normalization(x, name=name+'_norm')
        if activation:
            x = activation(x)
        return x


def conv2d_simple(x, n_out, ks, stride=1, padding='SAME', activation_fn=tf.nn.tanh,  name='conv2d', stddev=0.02):
    with tf.variable_scope(name):
        x = slim.conv2d(x, n_out, ks, stride, padding=padding, activation_fn = activation_fn,
                       weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        return x


def conv2d_transpose(x, n_out, ks, stride=1, padding='SAME', name='conv2d_transp', stddev=0.02, normalization=None):
    with tf.variable_scope(name):
        x = slim.conv2d_transpose(x, n_out, ks, stride, padding=padding, activation_fn = None,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        if normalization:
            x = normalization(x, name=name + '_norm')
        x = tf.nn.relu(x)
        return x


def res_block(input_x, ngf, ks=3, name='res_', normalization=None, dilation=1):
    p = int((ks - 1) / 2) + dilation - 1
    x = tf.pad(input_x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    x = conv2d(x, ngf, ks, 1, padding='VALID', name=name+'_c1', normalization=normalization, dilation=dilation)
    x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    x = conv2d(x, ngf, ks, 1, padding='VALID', name=name+'_c2', normalization=normalization,
               activation=None, dilation=dilation)
    return input_x + x



def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)
