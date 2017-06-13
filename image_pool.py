import random

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


