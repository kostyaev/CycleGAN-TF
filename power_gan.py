import tensorflow as tf
from losses import *
from networks import *

def convert2int(image):
  """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
  """
  return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def convert2float(image):
  """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return (image/127.5) - 1.0


def batch_convert2int(images):
  """
  Args:
    images: 4D float tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D int tensor
  """
  return tf.map_fn(convert2int, images, dtype=tf.uint8)

def batch_convert2float(images):
  """
  Args:
    images: 4D int tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D float tensor
  """
  return tf.map_fn(convert2float, images, dtype=tf.float32)

class PowerGAN:


    def __init__(self, name, img_size=None, ngf=32, ndf=32, input_ch=3, lambda_a=5, lambda_b=5, d_num_layers=5):
        criterion_gan = mae

        self.a_real = tf.placeholder(tf.float32,
                               [None, img_size, img_size, input_ch], name='A_real')
        self.b_real = tf.placeholder(tf.float32,
                                       [None, img_size, img_size, input_ch], name='B_real')

        self.fake_a_sample = tf.placeholder(tf.float32,
                                       [None, img_size, img_size, input_ch], name='fake_A_sample')

        self.fake_b_sample = tf.placeholder(tf.float32,
                                         [None, img_size, img_size, input_ch], name='fake_B_sample')

        G = PowerGenerator(ngf, name='G')

        # Generators
        self.A_fakeB = G(self.a_real)
        self.fake_B = self.A_fakeB[:, :, :, 3:]
        rec_A = self.A_fakeB[:, :, :, :3]


        self.fakeA_B = G(self.b_real)
        self.fake_A = self.fakeA_B[:, :, :, :3]
        rec_B = self.fakeA_B[:, :, :, 3:]


        DA = Discriminator(ndf, name='D_A', num_layers=d_num_layers)
        DB = Discriminator(ndf, name='D_B', num_layers=d_num_layers)

        #Discriminators
        DA_fake = DA(self.fake_A)
        DB_fake = DB(self.fake_B)


        # Generator Losses
        recon_loss = lambda_a * abs_criterion(self.a_real, rec_A) + lambda_b * abs_criterion(self.b_real, rec_B)
        self.g_loss = criterion_gan(DB_fake, 0.9) + criterion_gan(DA_fake, 0.9) + recon_loss


        DA_real = DA(self.a_real)
        DB_real = DB(self.b_real)
        DA_fake_sample = DA(self.fake_a_sample)
        DB_fake_sample = DB(self.fake_b_sample)


        # Discriminator Losses
        da_loss_real = criterion_gan(DA_real, 0.9)
        da_loss_fake = criterion_gan(DA_fake_sample, 0)
        self.da_loss = (da_loss_real + da_loss_fake) * 0.5

        db_loss_real = criterion_gan(DB_real, 0.9)
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

        self.G = G
        self.DA = DA
        self.DB = DB

        tf.summary.image('%s-A/original' % name, batch_convert2int(self.a_real))
        tf.summary.image('%s-A/generated' % name, batch_convert2int(self.fake_B))
        tf.summary.image('%s-A/reconstruction' % name, batch_convert2int(rec_A))

        tf.summary.image('%s-B/original' % name, batch_convert2int(self.b_real))
        tf.summary.image('%s-B/generated' % name, batch_convert2int(self.fake_A))
        tf.summary.image('%s-B/reconstruction' % name, batch_convert2int(rec_B))


    def get_losses(self):
        return self.g_loss, self.da_loss, self.db_loss










