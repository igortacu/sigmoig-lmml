#!/usr/bin/env python3
"""
export_weights.py
Rebuild model from train.py, load the checkpoint created by the callback,
and write a plain HDF5 file with weights only, named task_61.h5.
"""

import tensorflow as tf
from tensorflow import keras
import h5py

CLASSES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

def build_model():
    # same as in train.py
    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,
    )
    base_model = tf.keras.Model(
        base_model.inputs,
        outputs=[base_model.get_layer("conv2_block3_out").output],
    )

    inputs = keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x[0])
    x = keras.layers.Dense(len(CLASSES))(x)
    model = keras.Model(inputs, x)
    return model

def main():
    model = build_model()

    # load the good file written by the callback
    model.load_weights("best_model.weights.h5")

    # write only raw weights to task_61.h5
    with h5py.File("task_61.h5", "w") as f:
        grp = f.create_group("weights")
        for w in model.weights:
            # w.name looks like 'conv1_conv/kernel:0'
            ds_name = w.name.replace(":", "_")
            grp.create_dataset(ds_name, data=w.numpy())

    print("task_61.h5 written")

if __name__ == "__main__":
    main()
