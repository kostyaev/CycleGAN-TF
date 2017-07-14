from losses import *
from networks import *

def preprocess_image(im):
    im = tf.to_float(im)
    im = (im / 127.5) - 1.0
    return im

def postprocess_image(image, name=None):
    im = (image + 1) * 127.5
    im = tf.cast(im, tf.uint8, name=name)
    return im

class CycleGAN:

    def __init__(self, name, img_size=600, ngf=32, ndf=32, ks=5, input_ch=3,
                 lambda_a=10, lambda_b=10, d_num_layers=3, normalization='instance', training=True, dilated=False):
        criterion_gan = mae

        if normalization == 'instance':
            norm = instance_norm
        elif normalization == 'batch':
            norm = BatchNorm(is_training=training)
        else:
            norm = None

        self.input_a = tf.placeholder(tf.uint8,
                               [None, img_size, img_size, input_ch], name='A_real')

        self.input_b = tf.placeholder(tf.uint8,
                               [None, img_size, img_size, input_ch], name='B_real')

        self.input_fake_a_sample = tf.placeholder(tf.uint8,
                                       [None, img_size, img_size, input_ch], name='fake_A_sample')

        self.input_fake_b_sample = tf.placeholder(tf.uint8,
                                       [None, img_size, img_size, input_ch], name='fake_B_sample')

        self.a_real = preprocess_image(self.input_a)
        self.b_real = preprocess_image(self.input_b)

        self.fake_a_sample = preprocess_image(self.input_fake_a_sample)
        self.fake_b_sample = preprocess_image(self.input_fake_b_sample)

        GA = Generator(ngf, name='G_A', activation=tf.nn.tanh, ks=ks, norm=norm, dilation=dilated)
        GB = Generator(ngf, name='G_B', activation=tf.nn.tanh, ks=ks, norm=norm, dilation=dilated)

        #Generators
        self.fake_B = GA(self.a_real)
        rec_A = GB(self.fake_B)
        self.fake_A = GB(self.b_real)
        rec_B = GA(self.fake_A)

        DA = Discriminator(ndf, name='D_A', num_layers=d_num_layers, norm=norm)
        DB = Discriminator(ndf, name='D_B', num_layers=d_num_layers, norm=norm)

        #Discriminators
        DA_fake = DA(self.fake_A)
        DB_fake = DB(self.fake_B)

        #Generator Losses
        cycle_loss = lambda_a * abs_criterion(self.a_real, rec_A) + lambda_b * abs_criterion(self.b_real, rec_B)
        self.g_loss = criterion_gan(DB_fake, 1.0) + criterion_gan(DA_fake, 1.0) + cycle_loss

        #Reconstruction loss for background consistency
        self.rec_loss = lambda_a * abs_criterion(self.a_real, self.fake_B) + lambda_b * abs_criterion(self.b_real, self.fake_A)


        DA_real = DA(self.a_real)
        DB_real = DB(self.b_real)
        DA_fake_sample = DA(self.fake_a_sample)
        DB_fake_sample = DB(self.fake_b_sample)


        # Discriminator Losses
        da_loss_real = criterion_gan(DA_real, 1.0)
        da_loss_fake = criterion_gan(DA_fake_sample, 0)
        self.da_loss = (da_loss_real + da_loss_fake) * 0.5

        db_loss_real = criterion_gan(DB_real, 1.0)
        db_loss_fake = criterion_gan(DB_fake_sample, 0)
        self.db_loss = (db_loss_real + db_loss_fake) * 0.5


        self.g_loss_sum = tf.summary.scalar('G/loss', self.g_loss)

        self.da_loss_sum = tf.summary.scalar('DA/loss', self.da_loss)
        da_loss_real_sum = tf.summary.scalar('DA/loss_real', da_loss_real)
        da_loss_fake_sum = tf.summary.scalar('DA/loss_fake', da_loss_fake)

        self.db_loss_sum = tf.summary.scalar('DB/loss', self.db_loss)
        db_loss_real_sum = tf.summary.scalar('DB/loss_real', db_loss_real)
        db_loss_fake_sum = tf.summary.scalar('DB/loss_fake', db_loss_fake)

        self.da_sum = tf.summary.merge([self.da_loss_sum, da_loss_real_sum, da_loss_fake_sum])
        self.db_sum = tf.summary.merge([self.db_loss_sum, db_loss_real_sum, db_loss_fake_sum])

        self.GA = GA
        self.GB = GB
        self.DA = DA
        self.DB = DB

        tf.summary.image('%s-A/original' % name, postprocess_image(self.a_real), max_outputs=1)
        tf.summary.image('%s-A/generated' % name, postprocess_image(self.fake_B), max_outputs=1)
        tf.summary.image('%s-A/reconstruction' % name, postprocess_image(rec_A), max_outputs=1)

        tf.summary.image('%s-B/original' % name, postprocess_image(self.b_real), max_outputs=1)
        tf.summary.image('%s-B/generated' % name, postprocess_image(self.fake_A), max_outputs=1)
        tf.summary.image('%s-B/reconstruction' % name, postprocess_image(rec_B), max_outputs=1)

        self.fake_A_int = postprocess_image(self.fake_A, name='A_fake')
        self.fake_B_int = postprocess_image(self.fake_B, name='B_fake')


    def get_losses(self):
        return self.g_loss, self.da_loss, self.db_loss










