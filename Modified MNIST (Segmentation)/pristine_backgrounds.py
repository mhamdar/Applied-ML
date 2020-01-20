
import gc

import matplotlib.pyplot as plt
import pickle
import numpy as np

from skimage.morphology import dilation, label
from digit_extractor import *
from random import randrange

#pristine_backgrounds will right 100 files in a datater document.
#Each consists of ~1100 pristine backgrounds. 
#Toal of 110000 pristine backgrounds on which we can add numbers (see millions.py).


raw_images = pickle.load(open("data/train_max_x", "rb"))

num_splits = 100



for i, batch in enumerate(np.split(raw_images, num_splits)):
	pristine=[]
	for img in batch:
		pristine+=pristine_box(img)
	pickle.dump(pristine, open("datater/pristine_box_"+str(i)+".pkl", "wb"))

	print("Done with batch " + str(i + 1) + " of " + str(num_splits))
	
