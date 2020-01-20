import cv2
from albumentations import Compose, ToFloat, ShiftScaleRotate, ElasticTransform, OneOf, Lambda
from cv2.cv2 import GaussianBlur
from skimage.draw import ellipse
from tensorflow.python.keras.utils import Sequence
import numpy as np
import numpy.random as rd

import matplotlib.pyplot as plt

def randomBlotch(image, **kwargs):
    num_blotches = rd.randint(1, 3)
    blotch_intensity = 235 #brightness for blotch pixels

    canvas = np.zeros(image.shape)

    for i in range(num_blotches):
        ellipse_axes = (rd.randint(3, 7), rd.randint(3, 7))

        x_center = rd.randint(0, image.shape[0])
        y_center = rd.randint(0, image.shape[1])

        angle = rd.random_sample() * np.pi

        ellipse_coord = ellipse(x_center, y_center, ellipse_axes[0], ellipse_axes[1],
                                rotation=angle, shape=image.shape)

        canvas[ellipse_coord] = blotch_intensity

    # blurring canvas
    canvas = GaussianBlur(canvas, (3, 3), 1)

    #adding channel info
    canvas = np.reshape(canvas, image.shape)

    #casting to int
    canvas = canvas.astype(int)

    # applying canvas
    new_image = canvas + image

    #limiting brightness to 255
    new_image = np.vectorize(lambda x: x if x < 255 else 255)(new_image)

    return new_image


augmentations = Compose([
    ShiftScaleRotate(shift_limit=0.07,
                     scale_limit=0.07,
                     rotate_limit=50,
                     border_mode=cv2.BORDER_CONSTANT, value=0,
                     p=0.95),
    Lambda(image=randomBlotch, p=0.7),
    ToFloat(max_value=255)  # normalizes the data to [0,1] in float
])


class AugmentedDataSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size=32):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil((len(self.x) + 0.0) / self.batch_size))

    def __getitem__(self, index):
        x_batch = self.x[index * self.batch_size : (index + 1) * self.batch_size]
        y_batch = self.y[index * self.batch_size : (index + 1) * self.batch_size]

        augmented_image_batch =  [augmentations(image=im)["image"] for im in x_batch]
        return np.stack([a for a in augmented_image_batch]), np.array(y_batch)

