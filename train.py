from cycle_gan import *
from glob import glob
from data_loader import *
import functools
import sys
import tensorflow as tf
import time
import argparse
import os

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3


parser = argparse.ArgumentParser()

parser.add_argument("--name", help="Name of the experiment")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--dataset", help="path to folder containing images")
parser.add_argument("--checkpoint", default='checkpoints/', help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--tensorboard", default='tensorboard/', help='tensorboard dir')
parser.add_argument("--max_epochs", default=200, type=int, help="number of training epochs")

parser.add_argument("--ngf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=32, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--crop_size", type=int, default=256, help="crop size")
parser.add_argument("--batch_size", type=int, default=1, help='training batch size')
parser.add_argument("--save_freq", type=int, default=6000, help='Save checkpoint frequency')


args = parser.parse_args()


def log(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions[::-1], lambda x: x)


def random_subset(f_list, min_len=0, max_len=1):
    return np.random.choice(f_list, randint(min_len, max_len), replace=False)


def train(sess, data_dirs, epochs, start_lr=2e-4, beta1=0.5, checkpoints_dir='snapshots/', tensorboard_dir='tensorboard'):
    model = CycleGAN(name=args.name, lambda_a=5.0, lambda_b=5.0, ngf=args.ngf, ndf=args.ndf)
    g_loss, da_loss, db_loss = model.get_losses()

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    restorer = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=100)

    lr = tf.placeholder(tf.float32, shape=[], name="lr")

    da_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(da_loss, var_list=model.DA.variables)
    db_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(db_loss, var_list=model.DB.variables)
    g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(g_loss, var_list=[model.GA.variables + model.GB.variables])

    with tf.control_dependencies([g_optim, da_optim, db_optim]):
        optimizers = tf.no_op(name='optimizers')

    dataA = glob(data_dirs[0])
    dataB = glob(data_dirs[1])

    mirror_f = lambda x: compose(*random_subset([mirror], min_len=0, max_len=1))(x)
    crop_f = functools.partial(crop, crop_size=args.crop_size, center=False)
    resize_f = functools.partial(resize_aspect, min_px=args.crop_size, max_px=args.crop_size)

    train_pipeline = compose(load_image, mirror_f, resize_f, crop_f, img2array, preprocess)
    generatorA = batch_generator(lambda: image_generator(dataA, train_pipeline, shuffle=True), args.batch_size)
    generatorB = batch_generator(lambda: image_generator(dataB, train_pipeline, shuffle=True), args.batch_size)

    init = tf.global_variables_initializer()
    sess.run(init)

    ckpt = tf.train.latest_checkpoint(checkpoints_dir)
    if ckpt:
        log('Loading saved checkpoint: %s' % ckpt)
        restorer.restore(sess, ckpt)
        step = int(ckpt.split('-')[1])
    else:
        step = 0

    for epoch in range(1, epochs+1):
        log('starting epoch: %d' % epoch)
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

            if step % args.save_freq == 0:
                if not os.path.exists(checkpoints_dir):
                    os.makedirs(checkpoints_dir)
                save_path = saver.save(sess, checkpoints_dir + "/model_epoch_%d.ckpt" % epoch, global_step=step)
                log("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.set_random_seed(args.seed)

    trainA = os.path.join(args.dataset, 'trainA') + '/*.jpg'
    trainB = os.path.join(args.dataset, 'trainB') + '/*.jpg'

    tensorboard_dir = os.path.join(args.tensorboard, args.name)
    checkpoints_dir = os.path.join(args.checkpoint, args.name)

    print trainA, trainB

    with tf.Session() as sess:
        train(sess,
              data_dirs=[trainA, trainB],
              epochs=200,
              checkpoints_dir=checkpoints_dir,
              tensorboard_dir=tensorboard_dir)