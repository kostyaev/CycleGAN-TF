from ops import *



class Generator:
    def __init__(self, ngf, ks=3, name='generator', activation=tf.nn.tanh):
        self.name = name
        self.ks = ks
        self.ngf = ngf
        self.reuse = False
        self.activation = activation

    def __call__(self, image):
        with tf.variable_scope(self.name):
            if self.reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            pad = self.ks / 2
            x = tf.pad(image, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "REFLECT")
            x = conv2d(x, self.ngf, self.ks, 1, padding='VALID', name='g_c1')
            x = conv2d(x, self.ngf * 2, 3, 2, name='g_c2')
            x = conv2d(x, self.ngf * 4, 3, 2, name='g_c3')
            for i in range(4):
                x = res_block(x, self.ngf * 4, name='res%d_' % i)
            x = conv2d_transpose(x, self.ngf * 2, 3, 2, name='g_ct1')
            x = conv2d_transpose(x, self.ngf, 3, 2, name='g_ct2')
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "REFLECT")
            x = conv2d_simple(x, 3, self.ks, 1, padding='VALID', activation_fn=self.activation)
            if not self.activation:
                x = tf.clip_by_value(x, -1.0, 1.0)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return x


class Discriminator:
    def __init__(self, ndf, name='discriminator', num_layers=3):
        self.ndf = ndf
        self.name = name
        self.reuse = False
        self.num_layers = num_layers

    def __call__(self, image):
        ks = 4
        padding = 'SAME'
        with tf.variable_scope(self.name):
            if self.reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            x = slim.conv2d(image, self.ndf, ks, stride=2, padding=padding, activation_fn=lrelu)
            mult = 2
            for i in range(1, self.num_layers + 1):
                # stride = 2 if i % 2 == 1 else 1
                stride = 2
                x = slim.conv2d(x, self.ndf * mult, ks, stride=stride, padding=padding, activation_fn=None)
                x = instance_norm(x, '_norm_%d' % i)
                x = lrelu(x)
                mult *= stride
                mult = min(mult, 8)

            x = slim.conv2d(x, self.ndf * mult, ks, stride=1, padding=padding, activation_fn=None)
            x = instance_norm(x, '_norm_%d' % (self.num_layers + 1))
            x = lrelu(x)
            x = conv2d_simple(x, 1, ks, stride=1, padding=padding, activation_fn=None, name='out')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return x
