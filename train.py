from cycle_gan import *
from glob import glob
from data_loader import *
import functools
import sys
import tensorflow as tf
import time
import os

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3


def log(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions[::-1], lambda x: x)


def train(sess, data_dirs, epochs, start_lr=2e-4, beta1=0.5, checkpoints_dir='snapshots/', tensorboard_dir='tensorboard'):
    model = CycleGAN(lambda_a=10.0, lambda_b=10.0)
    g_loss, da_loss, db_loss = model.get_losses()

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=100)

    lr = tf.placeholder(tf.float32, shape=[], name="lr")

    da_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(da_loss, var_list=model.DA.variables)
    db_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(db_loss, var_list=model.DB.variables)
    g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(g_loss, var_list=[model.GA.variables + model.GB.variables])

    with tf.control_dependencies([g_optim, da_optim, db_optim]):
        optimizers = tf.no_op(name='optimizers')

    dataA = glob(data_dirs[0])
    dataB = glob(data_dirs[1])

    crop_f = functools.partial(crop, crop_size=256, center=True)
    resize_f = functools.partial(resize_aspect, maxPx=286, minPx=286)

    train_pipeline = compose(load_image, resize_f, crop_f, img2array, preprocess)
    generatorA = batch_generator(lambda:image_generator(dataA, train_pipeline, shuffle=False), 1)
    generatorB = batch_generator(lambda:image_generator(dataB, train_pipeline, shuffle=False), 1)

    ckpt = tf.train.get_checkpoint_state(checkpoints_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        log('Loading saved checkpoint: %s' % ckpt_name)
        saver.restore(sess, checkpoints_dir + ckpt_name)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    step = 0

    for epoch in range(1, epochs+1):
        if epoch < 100:
            curr_lr = start_lr
        else:
            curr_lr = start_lr - start_lr*(epoch-100)/100

        start_iter = time.time()
        for i in range(max(len(dataA), len(dataB))):
            step += 1

            batchA = generatorA.next()
            batchB = generatorB.next()

            input_real = {model.a_real: batchA, model.b_real: batchB}
            fakeA, fakeB = sess.run([model.fake_A, model.fake_B], input_real)

            _, lossG, lossDA, lossDB, summary = sess.run([optimizers, g_loss, da_loss, db_loss, summary_op],
                                                         {model.a_real: batchA, model.b_real: batchB,
                                                          model.fake_a_sample: fakeA, model.fake_b_sample: fakeB, lr: curr_lr})

            writer.add_summary(summary, step)
            writer.flush()


            if step % 50 == 0:
                end_iter = time.time()
                log('Step %d: G_loss: %.3f, DA_loss: %.3f, DB_loss: %.3f, time: %.3fs' % (step, lossG, lossDA, lossDB, end_iter - start_iter))
                start_iter = time.time()

            if step % 6000 == 0:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                log("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    tf.reset_default_graph()
    d_dir = '/root/storage/projects/CycleGAN/datasets/makeup_face_v2/{}/*.jpg'
    data_dirs = [d_dir.format('trainA'), d_dir.format('trainB')]

    checkpoints_dir = 'checkpoints'
    tensorboard_dir = '/root/storage/tensorboard/makeup_gan_256/'


    with tf.Session() as sess:
        train(sess, data_dirs=data_dirs, epochs=200, checkpoints_dir=checkpoints_dir, tensorboard_dir=tensorboard_dir)