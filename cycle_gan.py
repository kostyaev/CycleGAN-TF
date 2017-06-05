import tensorflow as tf
from losses import *
from ops import *

class CycleGAN:


    def __init__(self, img_size=128, input_ch=3, lambda_a=5, lambda_b=5, lr=2e-4, beta1=0.5):

        tf.reset_default_graph()

        criterion_gan = mae

        a_real = tf.placeholder(tf.float32,
                               [None, img_size, img_size, input_ch], name='A_real')
        b_real = tf.placeholder(tf.float32,
                                       [None, img_size, img_size, input_ch], name='B_real')

        fake_a_sample = tf.placeholder(tf.float32,
                                       [None, img_size, img_size, input_ch], name='fake_A_sample')

        fake_b_sample = tf.placeholder(tf.float32,
                                         [None, img_size, img_size, input_ch], name='fake_B_sample')

        # Generators
        fake_B = generator(a_real, 32, name='G_A2B', reuse=False)
        fake_fake_A = generator(fake_B, 32, name='G_B2A', reuse=False)
        fake_A = generator(b_real, 32, name='G_B2A', reuse=True)
        fake_fake_B = generator(fake_A, 32, name='G_A2B', reuse=True)


        #Discriminators
        DA_real = discriminator(a_real, 32, name='D_A', reuse=False)
        DB_real = discriminator(b_real, 32, name='D_B', reuse=False)

        DA_fake = discriminator(fake_A, 32, name='D_A', reuse=True)
        DB_fake = discriminator(fake_B, 32, name='D_B', reuse=True)
        DA_fake_sample = discriminator(fake_a_sample, 32, name='D_A', reuse=True)
        DB_fake_sample = discriminator(fake_b_sample, 32, name='D_B', reuse=True)


        # Generator Losses
        recon_loss = lambda_a * abs_criterion(a_real, fake_fake_A) + lambda_b * abs_criterion(b_real, fake_fake_B)
        g_loss_a2b = criterion_gan(DB_fake, tf.ones_like(DB_fake)*0.9)
        g_loss_b2a = criterion_gan(DA_fake, tf.ones_like(DA_fake)*0.9)
        self.g_loss = g_loss_a2b + g_loss_b2a + recon_loss


        # Discriminator Losses
        da_loss_real = criterion_gan(DA_real, tf.ones_like(DA_real)*0.9)
        da_loss_fake = criterion_gan(DA_fake_sample, tf.zeros_like(DA_fake_sample))
        self.da_loss = (da_loss_real + da_loss_fake) / 2

        db_loss_real = criterion_gan(DB_real, tf.ones_like(DB_real)*0.9)
        db_loss_fake = criterion_gan(DB_fake_sample, tf.zeros_like(DB_fake_sample))
        self.db_loss = (db_loss_real + db_loss_fake) / 2


        self.da_loss_sum = tf.summary.scalar('da_loss', self.da_loss)
        da_loss_real_sum = tf.summary.scalar('da_loss_real', da_loss_real)
        da_loss_fake_sum = tf.summary.scalar('da_loss_fake', da_loss_fake)

        self.db_loss_sum = tf.summary.scalar('db_loss', self.db_loss)
        db_loss_real_sum = tf.summary.scalar('db_loss_real', db_loss_real)
        db_loss_fake_sum = tf.summary.scalar('db_loss_fake', db_loss_fake)

        self.da_sum = tf.summary.merge([self.da_loss_sum, da_loss_real_sum, da_loss_fake_sum])
        self.db_sum = tf.summary.merge([self.db_loss_sum, db_loss_real_sum, db_loss_fake_sum])



    def initialize(self, sess):
        init_op = tf.global_variables_initializer()
        sess.run(init_op)


    def forward(self, sess):
        pass


    def train(self, sess, epochs, lr, beta1):
        writer = tf.summary.FileWriter("/root/storage/tensorboard/", sess.graph)
        counter = 0

        t_vars = tf.trainable_variables()
        db_vars = [var for var in t_vars if 'D_B' in var.name]
        da_vars = [var for var in t_vars if 'D_A' in var.name]
        g_vars_a2b = [var for var in t_vars if 'G_A2B' in var.name]
        g_vars_b2a = [var for var in t_vars if 'G_B2A' in var.name]
        g_vars = [var for var in t_vars if 'G_' in var.name]


        self.da_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(self.da_loss, var_list=da_vars)
        self.db_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(self.db_loss, var_list=db_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(self.g_loss, var_list=g_vars)





