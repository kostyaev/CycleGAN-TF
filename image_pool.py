import random

class ImagePool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.pool_images = []

    def query(self, image_pair):
        if len(self.pool_images) < self.pool_size:
            self.pool_images.append(image_pair)
            return image_pair
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self.pool_size - 1)
                temp = self.pool_images[random_id]
                self.pool_images[random_id] = image_pair
                return temp
            else:
                return image_pair


