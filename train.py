import argparse
import functools
import os
import sys
import time
from glob import glob
from multiprocessing import Process, Queue
from cycle_gan import *
from data_loader import *
from image_pool import ImagePool
import signal

parser = argparse.ArgumentParser()

parser.add_argument("--name", help="Name of the experiment")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--ks", default=5, type=int, help='Kernel size for generator')
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
parser.add_argument("--pool_size", type=int, default=50, help='Pool size')
parser.add_argument('--multi_scale', action='store_true', default=False,  help='Enable multiscale mode')
parser.add_argument('--color_jitter', action='store_true', default=False,  help='Enable color jittering')
parser.add_argument('--dilated', action='store_true', default=False,  help='Enable dilated convolutions')
parser.add_argument("--normalization", type=str, choices=['none', 'instance', 'batch'], default='instance', help='Choose normalization type')
parser.add_argument("--test_size", type=int, default=256, help="test size")
parser.add_argument("--backgrounds", default=None, help="path to folder containing images of background")


args = parser.parse_args()


class ImagePrefetcher:

    def __init__(self, generator):
        self.generator = generator
        self.process = Process(target=self._start)
        self.queue = Queue(maxsize=30)
        self.process.start()


    def _start(self):
        for batch in self.generator:
            self.queue.put(batch)

    def get(self):
        return self.queue.get()



def log(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions[::-1], lambda x: x)


def random_subset(f_list, min_len=0, max_len=1):
    return np.random.choice(f_list, randint(min_len, max_len), replace=False)


def train(sess, data_dirs, epochs, start_lr=2e-4, beta1=0.5, checkpoints_dir='snapshots/', tensorboard_dir='tensorboard'):
    print 'Chosen %s normalization type' % args.normalization
    img_size = None if args.multi_scale else args.crop_size
    model = CycleGAN(name=args.name, lambda_a=10.0, lambda_b=10.0, ks=args.ks, img_size=img_size, batch_size=args.batch_size,
                     ngf=args.ngf, ndf=args.ndf, d_num_layers=args.d_num_layers,
                     normalization=args.normalization, dilated=args.dilated)
    g_loss, da_loss, db_loss = model.get_losses()
    rec_loss = model.rec_loss

    # Visualize test data
    test_input = tf.placeholder(tf.uint8, [None, None, None, 3], name='test_input')
    show_a2b = tf.summary.image('%s/A2B' % args.name, test_input, max_outputs=200)
    show_b2a = tf.summary.image('%s/B2A' % args.name, test_input, max_outputs=200)

    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    restorer = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=200)

    lr = tf.placeholder(tf.float32, shape=[], name="lr")

    da_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(da_loss, var_list=model.DA.variables)
    db_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(db_loss, var_list=model.DB.variables)
    g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(g_loss, var_list=[model.GA.variables + model.GB.variables])
    rec_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(rec_loss, var_list=[model.GA.variables + model.GB.variables])

    with tf.control_dependencies([g_optim, da_optim, db_optim]):
        optimizers = tf.no_op(name='optimizers')


    dataA = glob(data_dirs[0])
    dataB = glob(data_dirs[1])
    data_testA = glob(data_dirs[2])
    data_testB = glob(data_dirs[3])
    dataC = glob(data_dirs[4]) if data_dirs[4] is not None else []


    print 'DataA: %d, DataB: %d, DataC: %d' % (len(dataA), len(dataB), len(dataC))

    mirror_f = lambda x: compose(*random_subset([mirror], min_len=0, max_len=1))(x)

    ops = [load_image, mirror_f]

    if args.multi_scale:
        print "Multiscale mode enabled"
        crop_scales = [192, 256, 320, 384]
        resize_scales = [192, 256, 320, 384, 420]
        ops.append(functools.partial(random_resize_crop, crop_scales=crop_scales, resize_scales=resize_scales))
    else:
        crop_scales = [args.crop_size]
        ops.append(functools.partial(resize_aspect_random, min_px=args.crop_size, max_px=args.scale_size))
        ops.append(functools.partial(crop, crop_size=args.crop_size, center=False))

    scale2id = dict([(s, idx) for idx, s in enumerate(crop_scales)])


    if args.color_jitter:
        print "Color jittering mode enabled"
        contrast_f = functools.partial(contrast,     steps=[0.9, 1.0, 1.1])
        brightness_f = functools.partial(brightness, steps=[0.9, 1.0, 1.1])
        saturation_f = functools.partial(saturation, steps=[0.9, 1.0, 1.1])
        color_jitter = lambda x: compose(*random_subset([contrast_f, brightness_f, saturation_f], min_len=0, max_len=3))(x)
        ops.append(color_jitter)

    ops.extend([img2array, preprocess])
    train_pipeline = compose(*ops)
    test_pipeline = compose(load_image,
                            functools.partial(resize_aspect, min_px=args.test_size, max_px=args.test_size),
                            functools.partial(crop, crop_size=args.test_size, center=True),
                            img2array)

    test_imagesA = list(image_generator(data_testA, test_pipeline, shuffle=False))
    test_imagesB = list(image_generator(data_testB, test_pipeline, shuffle=False))
    print 'Test data A: %d, test data B: %d' % (len(test_imagesA), len(test_imagesB))

    generatorA = ImagePrefetcher(batch_generator(lambda: image_generator(dataA, train_pipeline, shuffle=True), args.batch_size))
    generatorB = ImagePrefetcher(batch_generator(lambda: image_generator(dataB, train_pipeline, shuffle=True), args.batch_size))
    generatorC = ImagePrefetcher(batch_generator(lambda: image_generator(dataC, train_pipeline, shuffle=True), args.batch_size))

    fake_pools_A = [ImagePool(args.pool_size) for i in range(len(crop_scales))]
    fake_pools_B = [ImagePool(args.pool_size) for i in range(len(crop_scales))]


    def query(pools, img):
        idx = scale2id[img.shape[1]]
        return pools[idx].query(img)


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

    lossRec = 0.0

    for epoch in range(last_epoch, epochs+1):
        if epoch < args.decay_after:
            curr_lr = start_lr
        else:
            curr_lr = start_lr - start_lr*(epoch-args.decay_after)/args.decay_after

        log('starting epoch %d, lr: %f' % (epoch, curr_lr))


        start_iter = time.time()
        for i in range(iters_per_epoch):
            step += 1

            batchA = generatorA.get()
            batchB = generatorB.get()

            input_real = {model.a_real: batchA, model.b_real: batchB}
            fakeA, fakeB = sess.run([model.fake_A, model.fake_B], input_real)

            fake_a_sample, fake_b_sample = query(fake_pools_A, fakeA), query(fake_pools_B, fakeB)

            ops = [optimizers, g_loss, da_loss, db_loss]

            if i % args.display_freq == 0:
                _, lossG, lossDA, lossDB, summary = sess.run(ops + [model.summary],
                                                             {model.a_real: batchA, model.b_real: batchB,
                                                            model.fake_a_sample: fake_a_sample, model.fake_b_sample: fake_b_sample, lr: curr_lr})
                writer.add_summary(summary, step)
                writer.flush()
            else:
                _, lossG, lossDA, lossDB = sess.run(ops,
                                                    {model.a_real: batchA, model.b_real: batchB,
                                                     model.fake_a_sample: fake_a_sample, model.fake_b_sample: fake_b_sample, lr: curr_lr})

            # Train background reconstruction
            if len(dataC) > 0 and step % 10 == 0:
                batchC = generatorC.get()
                if batchC.shape[-1] == 3:
                    _, lossRec = sess.run([rec_optim, rec_loss], {model.a_real: batchC, model.b_real: batchC, lr: curr_lr})


            if step % 50 == 0:
                end_iter = time.time()
                log('Step %d: G_loss: %.3f, Rec_loss: %.3f, DA_loss: %.3f, DB_loss: %.3f, time: %.3fs' % (step, lossG, lossRec, lossDA, lossDB, end_iter - start_iter))
                start_iter = time.time()


            def save_images(images, is_a2b=True):
                result = []
                if is_a2b:
                    out_ = model.fake_B_int
                    in_ = model.input_a
                else:
                    out_ = model.fake_A_int
                    in_ = model.input_b
                for real in images:
                    fake = sess.run(out_, {in_: real})
                    collage = np.hstack((real[0], fake[0]))
                    result.append(collage)
                op = show_a2b if is_a2b else show_b2a
                summ = sess.run(op, {test_input: np.array(result)})
                writer.add_summary(summ, step)
                writer.flush()


            if step % args.save_freq == 0:
                log("Testing model...")
                save_images(test_imagesA, is_a2b=True)
                save_images(test_imagesB, is_a2b=False)
                if not os.path.exists(checkpoints_dir):
                    os.makedirs(checkpoints_dir)
                save_path = saver.save(sess, checkpoints_dir + "/model_epoch_%d.ckpt" % epoch, global_step=step)
                log("Model saved in file: %s" % save_path)


class SIGINT_handler():
    def __init__(self, session):
        self.SIGINT = False
        self.session = session

    def signal_handler(self, signal, frame):
        print('You pressed Ctrl+C! Exiting...')
        self.SIGINT = True
        self.session.close()
        sys.exit()


if __name__ == '__main__':

    tf.reset_default_graph()
    tf.set_random_seed(args.seed)

    trainA = os.path.join(args.dataset, 'trainA') + '/*.jpg'
    trainB = os.path.join(args.dataset, 'trainB') + '/*.jpg'
    testA = os.path.join(args.dataset, 'testA') + '/*.jpg'
    testB = os.path.join(args.dataset, 'testB') + '/*.jpg'
    trainC = args.backgrounds + '/*.jpg' if args.backgrounds is not None else None

    tensorboard_dir = os.path.join(args.tensorboard, args.name)
    checkpoints_dir = os.path.join(args.checkpoint, args.name)

    print trainA, trainB

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(config=config) as sess:
        handler = SIGINT_handler(session=sess)
        signal.signal(signal.SIGINT, handler.signal_handler)
        train(sess,
              data_dirs=[trainA, trainB, testA, testB, trainC],
              epochs=args.max_epochs,
              checkpoints_dir=checkpoints_dir,
              tensorboard_dir=tensorboard_dir)