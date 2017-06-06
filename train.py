import tensorflow as tf
from cycle_gan import *


def train(self, sess, epochs, lr=2e-4, beta1=0.5, checkpoints_dir='snapshots/'):
    model = CycleGAN()
    g_loss, da_loss, db_loss = model.get_losses()

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(checkpoints_dir, sess.graph)
    saver = tf.train.Saver()

    counter = 0

    da_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(da_loss, var_list=model.DA.variables)
    db_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(db_loss, var_list=model.DB.variables)
    g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(g_loss, var_list=[model.GA.variables + model.GB.variables])


    batchA = None
    batchB = None