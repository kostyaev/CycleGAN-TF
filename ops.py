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


def conv2d(x, n_out, ks, stride=1, padding='SAME', name='conv2d', stddev=0.02, activation=tf.nn.relu):
    with tf.variable_scope(name):
        x = slim.conv2d(x, n_out, ks, stride, padding=padding, activation_fn = None,
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        x = instance_norm(x, name=name+'_norm')
        if activation:
            x = activation(x)
        return x

def conv2d_simple(x, n_out, ks, stride=1, padding='SAME', activation_fn=tf.nn.tanh,  name='conv2d', stddev=0.02):
    with tf.variable_scope(name):
        x = slim.conv2d(x, n_out, ks, stride, padding=padding, activation_fn = activation_fn,
                       weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        return x

def conv2d_transpose(x, n_out, ks, stride=1, padding='SAME', name='conv2d_transp', stddev=0.02):
    with tf.variable_scope(name):
        x = slim.conv2d_transpose(x, n_out, ks, stride, padding=padding, activation_fn = None,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        x = instance_norm(x, name=name+'_norm')
        x = tf.nn.relu(x)
        return x

def res_block(input_x, ngf, ks=3, name='res_'):
    p = int((ks - 1) / 2)
    x = tf.pad(input_x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    x = conv2d(x, ngf, ks, 1, padding='VALID', name=name+'_c1')
    x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    x = conv2d(x, ngf, ks, 1, padding='VALID', name=name+'_c2', activation=None)
    return input_x + x



def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)
