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

class CycleGAN:


    def __init__(self, img_size=128, input_ch=3, lambda_a=5, lambda_b=5, lr=2e-4, beta1=0.5):
        criterion_gan = mae

        self.a_real = tf.placeholder(tf.float32,
                               [None, img_size, img_size, input_ch], name='A_real')
        self.b_real = tf.placeholder(tf.float32,
                                       [None, img_size, img_size, input_ch], name='B_real')

        self.fake_a_sample = tf.placeholder(tf.float32,
                                       [None, img_size, img_size, input_ch], name='fake_A_sample')

        self.fake_b_sample = tf.placeholder(tf.float32,
                                         [None, img_size, img_size, input_ch], name='fake_B_sample')

        GA = Generator(32, name='G_A')
        GB = Generator(32, name='G_B')

        # Generators
        self.fake_B = GA(self.a_real)
        fake_fake_A = GB(self.fake_B)
        self.fake_A = GB(self.b_real)
        fake_fake_B = GA(self.fake_A)

        DA = Discriminator(32, name='D_A')
        DB = Discriminator(32, name='D_B')

        #Discriminators
        DA_real = DA(self.a_real)
        DB_real = DB(self.b_real)

        DA_fake = DA(self.fake_A)
        DB_fake = DB(self.fake_B)
        DA_fake_sample = DA(self.fake_a_sample)
        DB_fake_sample = DB(self.fake_b_sample)


        # Generator Losses
        recon_loss = lambda_a * abs_criterion(self.a_real, fake_fake_A) + lambda_b * abs_criterion(self.b_real, fake_fake_B)
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

        tf.summary.image('A/generated', batch_convert2int(self.fake_B))
        tf.summary.image('A/reconstruction', batch_convert2int(fake_fake_A))
        tf.summary.image('B/generated', batch_convert2int(self.fake_A))
        tf.summary.image('B/reconstruction', batch_convert2int(fake_fake_B))


    def get_losses(self):
        return self.g_loss, self.da_loss, self.db_loss










