import os
import numpy as np
import sys
from SE_Inception_resnet_v2 import SE_Inception_resnet_v2
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
import pandas as pd
import cv2
from tqdm import tqdm
from collections import OrderedDict

from generator import DataGenerator
from utils import even_separate


def TTA(sess, test_lists, dir_path, ckpt_dir, augment_times=1):
    probs = np.zeros((augment_times, len(test_lists), len(category_df)))
    # ckpt config
    ckpt = tf.train.get_checkpoint_state(ckpt_dir )
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    for t in range(augment_times):
        test_datagen = DataGenerator(augment=True, random_erasing=True, horizontal_flip=True)
        test_generator = test_datagen.flow_from_list_prediction(
            lists=test_lists,
            batch_size=1,
            image_size=336,
            dir_path=dir_path)
        _probs = np.zeros((len(test_lists), len(category_df)))
        print(t+1, ": times")
        for i, v in tqdm(enumerate(test_lists)):
            inputs = next(test_generator)
            _prob = sess.run([prob], feed_dict={x: inputs, training_flag: False})
            _probs[i, :] = np.asarray(_prob)
        probs[t, :, :] = _probs

    # create pseudo_probs and predictions
    pseudo_probs = np.zeros((len(test_lists), len(category_df)))
    predictions = []
    for i in enumerate(range(probs.shape[1])):
        pseudo_prob = np.mean(probs[:, i, :], axis=0)
        pseudo_probs[i, :] = pseudo_prob
        predictions.append(np.argmax(pseudo_prob))

    predictions = np.asarray(predictions)
    return pseudo_probs, predictions

category_df = pd.read_csv("../dataset/cookpad/master.tsv", header=None)
test_lists = os.listdir("../dataset/cookpad/test")
image_size = 336
img_channels = 3
nb_classes = len(category_df)
dir_path = "../dataset/cookpad/test/"
ckpt_dir = "./model/SE_Inception_resnet_v2_1"
save_tsv ="./submit/submit_1.tsv"
save_probs = "./pseudo_probs/pseudo_prob_1.npy"
augment_times = 10

# build graph
# placeholders
x = tf.placeholder(
        tf.float32,
        shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(
        tf.float32,
        shape=[None, nb_classes])
training_flag = tf.placeholder(tf.bool)

logits = SE_Inception_resnet_v2(
        x,
        nb_classes,
        training=training_flag,
        ratio=4).model
prob = tf.nn.softmax(logits)

# trained model saver
saver = tf.train.Saver(tf.global_variables())

# session config
gpu_options = tf.GPUOptions(
        visible_device_list="3",
        allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)

# start session
with tf.Session(config=sess_config) as sess:
    pseudo_probs, predictions = TTA(sess, test_lists, dir_path, ckpt_dir, augment_times=augment_times)


np.save(save_probs, pseudo_probs)
dic = OrderedDict()
dic["file_name"] = np.asarray(test_lists)
dic["prediction"] = predictions
submit = pd.DataFrame(dic)
submit.to_csv(save_tsv, index=None, header=None, sep="\t")
