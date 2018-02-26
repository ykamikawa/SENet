# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize


def horizontal_flip(image, rate=0.5):
        if np.random.rand() < rate:
            image = image[::-1, :, :]
        image = image[:, ::-1, :]
        return image


def vertical_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
    return image


def random_rotation(image, angle_range=(0, 180)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = imresize(image, (h, w))
    return image


def random_crop(image, crop_size=(224, 224)):
    h, w, _ = image.shape

    # 0~(400-224)の間で画像のtop, leftを決める
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    # top, leftから画像のサイズである224を足して、bottomとrightを決める
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    image = image[top:bottom, left:right, :]
    return image


def scale_augmentation(image, scale_range=(256, 400), crop_size=224):
    scale_size = np.random.randint(*scale_range)
    image = imresize(image, (scale_size, scale_size))
    image = random_crop(image, (crop_size, crop_size))
    return image


class DataGenerator():
    def __init__(self):
        self.reset()

    def reset(self):
        self.images = []
        self.labels = []

    def flow_from_dataframe(self, df, nb_classes, batch_size, image_size, augment=True):
        dir_path = "../dataset/cookpad/train/"
        while True:
            if len(self.images):
                inputs = np.asarray(self.images)
                targets = np.asarray(self.labels)
                self.reset()

                yield inputs, targets

            for i, row in df.iterrows():
                img = cv2.imread(dir_path + row.file_name)[:, :, ::-1]
                img = cv2.resize(img, (image_size, image_size))
                array_img = img_to_array(img)/255

                if augment:
                    array_img = horizontal_flip(array_img)
                    array_img = vertical_flip(array_img)
                self.images.append(array_img)

                label = row.category_id
                label = np_utils.to_categorical(label, num_classes=nb_classes)
                label = np.reshape(label, [nb_classes])
                self.labels.append(label)

                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images)
                    targets = np.asarray(self.labels)
                    self.reset()

                    yield inputs, targets
