# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize


class DataGenerator():
    def __init__(
            self,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.5,
            zoom_range=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            random_erasing=False,
            mixup=False,
            mixup_alpha=0.2,
            augment=False):

        self.datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=0,
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                channel_shift_range=False,
                fill_mode="nearest",
                cval=0.,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                rescale=None,
                preprocessing_function=None,
                data_format=None)
        self.random_erasing = random_erasing
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.augment = augment
        self.reset()

    def reset(self):
        self.images = []
        self.labels = []

    # todo
    def _mixup(self, x_batch, y_batch):
        batch_size, h, w, c = x_batch.shape
        _, nb_classes = y_batch.shape
        l = np.random.beta(self.mixup_alpha, self.mixup_alpha, batch_size)
        x1 = x_batch[:int(batch_size/2)]
        x2 = x_batch[int(batch_size/2):]
        y1 = y_batch[:int(batch_size/2)]
        y2 = y_batch[int(batch_size/2):]
        x_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)

        x = x1 * x_l + x2 * (1 - x_l)
        y = y1 * y_l + y2 * (1 - y_l)

        return x, y

    def _random_erasing(self, image_org, p=0.5, s=(0.02, 0.4), r=(0.3, 3)):
        # Whether to process
        if np.random.rand() > p:
            return image_org
        image = np.copy(image_org)

        # Random pixel value to be masked
        mask_value = np.random.randint(0, 256)

        h, w, _ = image.shape
        # Random mask size
        mask_area = np.random.randint(h * w * s[0], h * w * s[1])

        # Random mask aspect ratio
        mask_aspect_ratio = np.random.rand() * r[1] + r[0]

        # determine mask width and height
        mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
        if mask_height > h - 1:
            mask_height = h - 1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w - 1:
            mask_width = w - 1

        top = np.random.randint(0, h - mask_height)
        left = np.random.randint(0, w - mask_width)
        bottom = top + mask_height
        right = left + mask_width
        image[top:bottom, left:right, :].fill(mask_value)
        return image


    def flow_from_dataframe(self, df, nb_classes, batch_size, image_size, dir_path):
        while True:
            if len(self.images):
                inputs = np.asarray(self.images)
                targets = np.asarray(self.labels)
                self.reset()
                if self.mixup:
                    inputs, targets = self._mixup(inputs, targets)

                yield inputs, targets

            for i, row in df.iterrows():
                img = cv2.imread(dir_path + row.file_name)[:, :, ::-1]
                img = cv2.resize(img, (image_size, image_size))
                array_img = img_to_array(img)

                if self.augment:
                    array_img = self.datagen.random_transform(array_img)
                if self.random_erasing:
                    array_img = self._random_erasing(array_img)
                self.images.append(array_img/255)

                label = int(row.category_id)
                label = np_utils.to_categorical(label, num_classes=nb_classes)
                label = np.reshape(label, [nb_classes])
                self.labels.append(label)

                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images)
                    targets = np.asarray(self.labels)
                    self.reset()
                    if self.mixup:
                        inputs, targets = self._mixup(inputs, targets)

                    yield inputs, targets
