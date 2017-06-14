import random
import numpy as np

class ImagePool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.pool_images = []

    def query(self, image):
        if len(self.pool_images) < self.pool_size:
            self.pool_images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self.pool_size - 1)
                temp = self.pool_images[random_id]
                self.pool_images[random_id] = image
                return temp
            else:
                return image



class BatchedImagePool:

    def __init__(self, pool_size=50, batch_size=2):
        self.pool_size = pool_size
        self.pool_images = []
        self.batch_size = batch_size
        self.range = range(pool_size)

    def get_random_with_replace(self, image):
        ids = np.random.choice(self.range, size=self.batch_size, replace=False)
        images = np.vstack([self.pool_images[i] for i in ids])
        if random.random() > 0.5:
            self.pool_images[ids[0]] = image
        else:
            images[0] = image
        return images

    def query(self, image):
        if len(self.pool_images) < self.pool_size:
            self.pool_images.append(image)
            return image
        else:
            return self.get_random_with_replace(image)