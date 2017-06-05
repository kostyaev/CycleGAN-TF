import tensorflow as tf
from tensorflow.contrib import slim


def instance_norm(input_var, epsilon=1e-5, axis = [1,2]):
    mean, var = tf.nn.moments(input_var, axis, keep_dims=True)
    return (input_var - mean) / tf.sqrt(var+epsilon)

def res_block(input_x, ngf, ks=3, name='res_'):
    p = int((ks - 1) / 2)
    x = tf.pad(input_x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    x = conv2d(x, ngf, ks, 1, padding='VALID', name=name+'_c1')
    x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    x = conv2d(x, ngf, ks, 1, padding='VALID', name=name+'_c2')
    return input_x + x

def conv2d(x, n_out, ks, stride=1, padding='SAME', name='conv2d', stddev=0.02):
    with tf.variable_scope(name):
        x = slim.conv2d(x, n_out, ks, stride, padding=padding, activation_fn = None,
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        x = instance_norm(x)
        x = tf.nn.relu(x)
        return x

def conv2d_simple(x, n_out, ks, stride=1, padding='SAME', activation_fn=tf.nn.tanh,  name='conv2d', stddev=0.02):
    with tf.variable_scope(name):
        x = slim.conv2d(x, n_out, ks, stride, padding=padding, activation_fn = None,
                       weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        return x

def conv2d_transpose(x, n_out, ks, stride=1, padding='SAME', name='conv2d_transp', stddev=0.02):
    with tf.variable_scope(name):
        x = slim.conv2d_transpose(x, n_out, ks, stride, padding=padding, activation_fn = None,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        x = instance_norm(x)
        x = tf.nn.relu(x)
        return x

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def generator(image, ngf, ks=3, name='generator', reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        x = tf.pad(image, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")
        x = conv2d(x, ngf, 5, 1, padding='VALID', name='g_c1')
        x = conv2d(x, ngf*2, 3, 2, name='g_c2')
        x = conv2d(x, ngf*4, 3, 2, name='g_c3')
        for i in range(4):
            x = res_block(x, ngf*4, name='res%d_'%i)
        x = conv2d_transpose(x, ngf*2, 3, 2, name='g_ct1')
        x = conv2d_transpose(x, ngf, 3, 2, name='g_ct2')
        x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")
        x = conv2d_simple(x, 3, 5, 1, padding='VALID', activation_fn=tf.nn.tanh, name='out')
    return x

def discriminator(image, ndf, name='discriminator', reuse=False):
    ks = 4
    padding = 'SAME'
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        x = slim.conv2d(image, ndf, ks, stride=2, padding=padding, activation_fn = lrelu)
        mult = 1
        for i in range(1):
            mult *= 2
            x = slim.conv2d(x, ndf*mult, ks, stride=2, padding=padding, activation_fn = None)
            x = instance_norm(x)
            x = lrelu(x)
        mult *= 2
        x = slim.conv2d(x, ndf*mult, ks, stride=1, padding=padding, activation_fn = None)
        x = instance_norm(x)
        x = lrelu(x)
        x = conv2d_simple(x, ndf*mult, ks, stride=1, padding=padding, activation_fn = None, name='out')
    return x