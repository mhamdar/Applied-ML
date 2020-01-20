import pickle
import numpy as np
from keras import Model

from keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, concatenate, Concatenate, Flatten

import pandas as pd
from keras.utils import plot_model, to_categorical

# Layers properties for raw image input
RES_BLOCK_LENGTH_RAW = 2  # Amount of CNN layers before pooling / res occurs for raw images
RES_BLOCK_NUM_RAW = 3  # Amount of res blocks
CNN_WIDTH_RAW = 64  # Width of each CNN layer
CNN_FIRST_WIDTH_RAW = 32

# Layers properties for the segmented digits input
RES_BLOCK_LENGTH_SEGMENTED = 2
RES_BLOCK_NUM_SEGMENTED = 2
CNN_WIDTH_SEGMENTED = 64

MERGE_LAYER_WIDTH = 100  # Width of Dense layer used to merge the different inputs

DENSE_LAYERS_NUM = 3  # Amount of final dense layers
DENSE_LAYERS_WIDTH = 120  # Width of each such dense layer
DROPOUT = 0.5  # Dropout proportion

OUTPUT_CATEGORIES = 10  # Amount of desired outputs

def load_segmented_digits(names):
    segmented_digits = []
    for filename in names:
        file = open(filename, "rb")
        segmented_digits += pickle.load(file)

        file.close()

    return segmented_digits


# Model builder
def build_model():
    raw_image_input = Input(shape=(128, 128, 1), name="raw_image_input")
    segmented_digit_input = [Input(shape=(28, 28, 1), name="seg_digit_" + str(i) + "_input") for i in range(3)]

    # building resnet section for raw image input

    previous_layer = Conv2D(CNN_FIRST_WIDTH_RAW, kernel_size=5, padding="same", name="raw_layer_1")(raw_image_input)
    previous_layer = Conv2D(CNN_FIRST_WIDTH_RAW, kernel_size=5, padding="same", name="raw_layer_2")(previous_layer)
    previous_layer = MaxPooling2D(pool_size=(2,2), name="raw_first_pooling")(previous_layer)

    res_tails = [previous_layer]

    for i in range(RES_BLOCK_NUM_RAW):
        if i >= 1: # after the first pooling block
            previous_layer = concatenate([previous_layer, res_tails[i-1]])

        for j in range(RES_BLOCK_LENGTH_RAW):
            previous_layer = Conv2D(CNN_WIDTH_RAW, kernel_size=3, padding="same")(previous_layer)

        if i != RES_BLOCK_NUM_RAW - 1: #final run
            previous_layer = Conv2D(CNN_FIRST_WIDTH_RAW, kernel_size=1, padding="same")(previous_layer)

        else:
            previous_layer = Conv2D(8, kernel_size=1, padding="same")(previous_layer)


        res_tails.append(previous_layer)

    previous_layer = MaxPooling2D(pool_size=(4,4))(previous_layer)
    previous_layer = Flatten()(previous_layer)
    raw_input_tail = Dropout(DROPOUT)(previous_layer)

    segmented_input_tails = []


    # building resnet section for segmented digit input
    for i in range(3):  # for every digit
        previous_layer = segmented_digit_input[i]

        for j in range(RES_BLOCK_NUM_SEGMENTED):
            for k in range(RES_BLOCK_LENGTH_SEGMENTED):
                previous_layer = Conv2D(CNN_WIDTH_SEGMENTED, kernel_size=3, padding="same")(previous_layer)

            if j != RES_BLOCK_NUM_SEGMENTED - 1:
                previous_layer = MaxPooling2D()(previous_layer)

            else:
                previous_layer = MaxPooling2D(pool_size=(4,4))(previous_layer)

        previous_layer = Flatten()(previous_layer)
        segmented_input_tail = Dropout(DROPOUT)(previous_layer)
        segmented_input_tails.append(segmented_input_tail)
        #building dense layers at the tail


    #merging both networks from either input
    previous_layer = concatenate(segmented_input_tails + [raw_input_tail])

    for i in range(DENSE_LAYERS_NUM):
        previous_layer = Dense(DENSE_LAYERS_WIDTH)(previous_layer)

    last_hidden_layer = previous_layer

    output = Dense(OUTPUT_CATEGORIES, activation="softmax")(last_hidden_layer)

    model = Model(inputs=[raw_image_input] + segmented_digit_input, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


raw_train_file = open("data/train_max_x", "rb")
raw_images = pickle.load(raw_train_file)
raw_train_file.close()
raw_images = np.reshape(np.array(raw_images), (50000, 128, 128, 1))

segmented_digits = load_segmented_digits(["data/train_segmented_digits_" + str(i) + ".pkl" for i in range(100)])

first_digit = np.array([example[0] for example in segmented_digits])
second_digit = np.array([example[1] for example in segmented_digits])
third_digit = np.array([example[2] if len(example) >= 3 else np.zeros((28, 28)) for example in segmented_digits])

first_digit = np.reshape(first_digit, (50000, 28, 28, 1))
second_digit = np.reshape(second_digit, (50000, 28, 28, 1))
third_digit = np.reshape(third_digit, (50000, 28, 28, 1))

df = pd.read_csv("data/train_max_y.csv")
y_train = list(df["Label"])
y_train = to_categorical(y_train)

model = build_model()

if __name__ == "__main__":
    model.fit([raw_images, first_digit, second_digit, third_digit],
              [y_train],
              epochs=3,
              batch_size=128
              )

