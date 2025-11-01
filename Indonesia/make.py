import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 224
MAX_EPS = 0.2
STEP = 0.01
ITERS = 120
TARGET_NAME = "giant panda"   # adjust below

# 1. surrogate from TF hub (here: imagenet-like)
base_model = keras.applications.MobileNetV2(weights="imagenet", include_top=True)
preprocess = keras.applications.mobilenet_v2.preprocess_input
decode = keras.applications.mobilenet_v2.decode_predictions

PANDA_INDEX = 388   # ImageNet: 388 = giant panda

# 2. load your source image
img = keras.preprocessing.image.load_img("input.jpg", target_size=(IMG_SIZE, IMG_SIZE))
img = keras.preprocessing.image.img_to_array(img)
img0 = img / 255.0
img_tf = tf.constant(img0, dtype=tf.float32)

pert = tf.Variable(tf.zeros_like(img_tf))

for i in range(ITERS):
    with tf.GradientTape() as tape:
        tape.watch(pert)
        adv = tf.clip_by_value(img_tf + pert, 0.0, 1.0)
        # MobileNetV2 expects [-1,1]
        adv_in = preprocess(adv * 255.0)[None, ...]
        logits = base_model(adv_in, training=False)[0]
        target = tf.one_hot(PANDA_INDEX, logits.shape[-1])
        loss = -tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits)
    grads = tape.gradient(loss, pert)
    step = STEP * tf.sign(grads)
    pert.assign_add(step)
    pert.assign(tf.clip_by_value(pert, -MAX_EPS, MAX_EPS))

pert_np = pert.numpy().astype("float32")
np.save("perturbation.npy", pert_np)
print("saved perturbation.npy")
