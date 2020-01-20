%tensorflow_version 2.x

from tensorflow.python.keras.utils import Sequence
import numpy as np

import matplotlib.pyplot as plt

from keras.datasets import mnist
from random import randrange
from keras.utils import to_categorical

#Thruple_Background takes a list of pristine backgrounds as input. 
#And in a "keras.Sequence" fashion will give you batches of pristine 
# background and 3 digits positioned and rotated randomly with degrees 
#between -30 and 30 degrees. Per pristine background __getitem__ generates 
#10 different digit combinations. That value can be modified. 


class Thruple_Background(Sequence):
    #x_set is a set of pristine backgrounds
    def __init__(self, x_set, y_set=None, batch_size=32):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train_range=len(self.x_train)

    def __len__(self):
        return int(np.ceil((len(self.x) + 0.0) / self.batch_size))

    def __getitem__(self, index):
        x_batch = self.x[index * self.batch_size : (index + 1) * self.batch_size]

        augmented=  []
        y_batch=[]
        for x in x_batch:
            for i in range(1):
                rand1=randrange(0, self.x_train_range)
                digit1=self.x_train[rand1]
                result1=self.y_train[rand1]

                rand2=randrange(0, self.x_train_range)
                digit2=self.x_train[rand2]
                result2=self.y_train[rand2]

                rand3=randrange(0, self.x_train_range)
                digit3=self.x_train[rand3]
                result3=self.y_train[rand3]

                angle1=randrange(-30, 30)
                angle2=randrange(-30, 30)
                angle3=randrange(-30, 30)

                thruple=thruple_mnist(digit1, digit2, digit3, angle1=angle1, angle2=angle2, angle3=angle3)
                new_training_sample = combine_pristine_mnist(x, thruple)
                new_training_sample = np.reshape(new_training_sample, (128,128,1)) #adding fake color channel for CNN

                augmented.append(new_training_sample / 255.) #normalizing input
                #activation = np.zeros(10)
                #for result in [result1, result2, result3]:
                #  activation[result] += 1/3.
                #y_batch.append(activation)
                y_batch.append(max([result1, result2, result3]))
        return np.array(augmented), np.array(y_batch)
