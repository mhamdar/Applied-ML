
import gc

import matplotlib.pyplot as plt
import pickle
import numpy as np

from skimage.morphology import dilation, label
from digit_extractor import *
from random import randrange


raw_images = pickle.load(open("data/train_max_x", "rb"))

print("Done loading train images")

num_splits = 100

pristine=[]

for i, batch in enumerate(np.split(raw_images, num_splits)):
	pre_pristine=[pristine_box(img) for img in batch]
	pristine+=[reshape_pristine(prist) for prist in pre_pristine if min(prist.shape)>=28]
	print("Done with batch " + str(i + 1) + " of " + str(num_splits))

pickle.dump(pristine, open("databis/pristine.pkl", "wb"))

####################################################################
pristine=pickle.load(open("databis/pristine.pkl", "rb"))

print("done loading")

(x_mnist_train, y_mnist_train), (x_mnist_test, y_mnist_test)=mnist.load_data()

print("done loading mnist")
x_test=[]
y_test=[]

for prist in pristine:
    rand=randrange(len(x_mnist_test))
    x_test+=[combine_pristine_mnist(prist, x_mnist_test[rand])]
    y_test+=[y_mnist_test[rand]]

pickle.dump(x_test, open("datater/xtest.pkl", "wb"))
pickle.dump(y_test, open("datater/ytest.pkl", "wb"))



