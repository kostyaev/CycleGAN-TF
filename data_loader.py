import numpy as np
from PIL import Image
import math


def img2array(img):
    a = np.array(img)
    if len(a.shape) == 2:
        a = a[:, :, np.newaxis]
    return a


def preprocess(a):
    return a / 127.5 - 1


def postprocess(a):
    a = (a + 1) * 127.5
    return np.clip(a, 0, 255).astype(np.uint8)


def randint(a, b):
    """Returns random uniform value between a and b inclusively"""
    return np.random.randint(a, b + 1, 1)[0]


def load_image(path):
    return Image.open(path)


def resize_aspect(img, min_px=128, max_px=128):
    width = img.size[0]
    height = img.size[1]
    smallest = min(width, height)
    largest = max(width, height)
    k = 1
    if largest > max_px:
        k = max_px / float(largest)
        smallest *= k
        largest *= k
    if smallest < min_px:
        k *= min_px / float(smallest)
    size = int(math.ceil(width * k)), int(math.ceil(height * k))
    img = img.resize(size, Image.BILINEAR)
    return img


def resize(img, min_px=256, max_px=286):
    w = randint(min_px, max_px)
    h = randint(min_px, max_px)
    return img.resize((w, h), Image.BILINEAR)


def crop(img, crop_size=128, center=False):
    width, height = img.size
    if not center:
        h_off = randint(0, height - crop_size)
        w_off = randint(0, width - crop_size)
    else:
        h_off = (height - crop_size) / 2
        w_off = (width - crop_size) / 2
    return img.crop((w_off, h_off, w_off + crop_size, h_off + crop_size))


def mirror(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def image_generator(data, img_transform, shuffle=False):
    if shuffle:
        np.random.shuffle(data)
    for img in data:
        try:
            x = img_transform(img)
            yield x[np.newaxis,:]
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            raise e


def batch_generator(generator, batch_size=32):
    while True:
        X = []
        g = generator()
        for x in g:
            X.append(x[0])
            if len(X) == batch_size:
                yield np.array(X)
                X = []
