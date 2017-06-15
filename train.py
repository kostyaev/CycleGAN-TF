import argparse
import functools
import os
import sys
import time
from glob import glob
import tensorflow as tf
from cycle_gan import *
from data_loader import *
from image_pool import ImagePool, BatchedImagePool



parser = argparse.ArgumentParser()

parser.add_argument("--name", help="Name of the experiment")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--dataset", help="path to folder containing images")
parser.add_argument("--checkpoint", default='checkpoints/', help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--tensorboard", default='tensorboard/', help='tensorboard dir')
parser.add_argument("--max_epochs", default=100, type=int, help="number of training epochs")
parser.add_argument("--decay_after", default=50, type=int, help="number of epoch from which start to decay lr")

parser.add_argument("--ngf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=32, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping")
parser.add_argument("--crop_size", type=int, default=256, help="crop size")
parser.add_argument("--batch_size", type=int, default=1, help='training batch size')
parser.add_argument("--save_freq", type=int, default=6000, help='Save checkpoint frequency')
parser.add_argument("--d_num_layers", type=int, default=3, help='Number of layers in discriminator')
parser.add_argument("--display_freq", type=int, default=1000, help='Update tensorboard frequency')


args = parser.parse_args()


def log(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions[::-1], lambda x: x)


def random_subset(f_list, min_len=0, max_len=1):
    return np.random.choice(f_list, randint(min_len, max_len), replace=False)


def train(sess, data_dirs, epochs, start_lr=2e-4, beta1=0.5, checkpoints_dir='snapshots/', tensorboard_dir='tensorboard'):
    model = CycleGAN(name=args.name, lambda_a=10.0, lambda_b=10.0, ngf=args.ngf, ndf=args.ndf, d_num_layers=args.d_num_layers)
    g_loss, da_loss, db_loss = model.get_losses()
    rec_loss = model.rec_loss

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    restorer = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=100)

    lr = tf.placeholder(tf.float32, shape=[], name="lr")

    da_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(da_loss, var_list=model.DA.variables)
    db_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(db_loss, var_list=model.DB.variables)
    g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(g_loss, var_list=[model.GA.variables + model.GB.variables])
    rec_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(rec_loss, var_list=[model.GA.variables + model.GB.variables])

    with tf.control_dependencies([g_optim, da_optim, db_optim]):
        optimizers = tf.no_op(name='optimizers')

    dataA = glob(data_dirs[0])
    dataB = glob(data_dirs[1])
    dataC = glob(data_dirs[2])

    print 'DataA: %d, DataB: %d, DataC: %d' % (len(dataA), len(dataB), len(dataC))

    mirror_f = lambda x: compose(*random_subset([mirror], min_len=0, max_len=1))(x)
    crop_f = functools.partial(crop, crop_size=args.crop_size, center=False)
    resize_f = functools.partial(resize_aspect_random, min_px=args.crop_size, max_px=args.scale_size)

    contrast_f = functools.partial(contrast, steps=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    brightness_f = functools.partial(brightness, steps=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    saturation_f = functools.partial(saturation, steps=[0.8, 0.9, 1.0, 1.1, 1.2])

    color_jitter = lambda x: compose(*random_subset([contrast_f, brightness_f, saturation_f], min_len=0, max_len=3))(x)

    train_pipeline = compose(load_image, mirror_f, resize_f, crop_f, img2array, preprocess)

    generatorA = batch_generator(lambda: image_generator(dataA, train_pipeline, shuffle=True), args.batch_size)
    generatorB = batch_generator(lambda: image_generator(dataB, train_pipeline, shuffle=True), args.batch_size)
    generatorC = batch_generator(lambda: image_generator(dataC, train_pipeline, shuffle=True), args.batch_size)


    fake_poolA = ImagePool(50)
    fake_poolB = ImagePool(50)


    init = tf.global_variables_initializer()
    sess.run(init)

    ckpt = tf.train.latest_checkpoint(checkpoints_dir)
    if ckpt:
        log('Loading saved checkpoint: %s' % ckpt)
        restorer.restore(sess, ckpt)
        step = int(ckpt.split('-')[1])
    else:
        step = 0

    iters_per_epoch = max(len(dataA), len(dataB))
    last_epoch = step / iters_per_epoch + 1


    for epoch in range(last_epoch, epochs+1):
        if epoch < args.decay_after:
            curr_lr = start_lr
        else:
            curr_lr = start_lr - start_lr*(epoch-args.decay_after)/args.decay_after

        log('starting epoch %d, lr: %f' % (epoch, curr_lr))


        start_iter = time.time()
        for i in range(iters_per_epoch):
            step += 1

            batchA = generatorA.next()
            batchB = generatorB.next()

            input_real = {model.a_real: batchA, model.b_real: batchB}
            fakeA, fakeB = sess.run([model.fake_A, model.fake_B], input_real)

            fake_a_sample, fake_b_sample = fake_poolA.query(fakeA), fake_poolB.query(fakeB)

            ops = [optimizers, g_loss, da_loss, db_loss]

            if i % args.display_freq == 0:
                _, lossG, lossDA, lossDB, summary = sess.run(ops + [summary_op],
                                                             {model.a_real: batchA, model.b_real: batchB,
                                                              model.fake_a_sample: fake_a_sample, model.fake_b_sample: fake_b_sample, lr: curr_lr})
            else:
                _, lossG, lossDA, lossDB = sess.run(ops,
                                                    {model.a_real: batchA, model.b_real: batchB,
                                                     model.fake_a_sample: fake_a_sample, model.fake_b_sample: fake_b_sample, lr: curr_lr})

            # Train background reconstruction
            if step % 5 == 0:
                batchC = generatorC.next()
                if batchC.shape[-1] == 3:
                    _, recLoss = sess.run([rec_optim, rec_loss], {model.a_real: batchC, model.b_real: batchC, lr: curr_lr})


            writer.add_summary(summary, step)
            writer.flush()

            if step % 50 == 0:
                end_iter = time.time()
                log('Step %d: G_loss: %.3f, Rec_loss: %.3f, D_loss: %.3f, time: %.3fs' % (step, lossG, lossRec, lossD, end_iter - start_iter))
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
    trainC = '/root/storage/playbooks/data/image-turk/backgrounds/*.JPEG'

    tensorboard_dir = os.path.join(args.tensorboard, args.name)
    checkpoints_dir = os.path.join(args.checkpoint, args.name)

    print trainA, trainB, trainC

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(config=config) as sess:
        train(sess,
              data_dirs=[trainA, trainB, trainC],
              epochs=args.max_epochs,
              checkpoints_dir=checkpoints_dir,
              tensorboard_dir=tensorboard_dir)