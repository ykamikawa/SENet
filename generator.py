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
            rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            horizontal_flip=True,
            vertical_flip=True,
            random_crop=False,
            scale_augmentation=False,
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

        self.scale_augmentation = scale_augmentation
        self.random_crop = random_crop
        self.random_erasing = random_erasing
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.augment = augment
        self._reset()

    def _reset(self):
        self.images = []
        self.labels = []

    # todo
    def _mixup(self, x_batch, y_batch):
        batch_size, h, w, c = x_batch.shape
        harf_batch = int(batch_size/2)
        _, nb_classes = y_batch.shape
        l = np.random.beta(self.mixup_alpha, self.mixup_alpha, harf_batch)
        x_l = l.reshape(harf_batch, 1, 1, 1)
        y_l = l.reshape(harf_batch, 1)

        x_1 = x_batch[:harf_batch]
        y_1 = y_batch[:harf_batch]
        x_2 = x_batch[harf_batch:]
        y_2 = y_batch[harf_batch:]

        x_1 = x_1 * x_l + x_2 * (1 - x_l)
        y_1 = y_1 * y_l + y_2 * (1 - y_l)
        x_2 = x_2 * x_l + x_1 * (1 - x_l)
        y_2 = y_2 * y_l + y_1 * (1 - y_l)

        x = np.vstack((x_1, x_2))
        y = np.vstack((y_1, y_2))

        return x, y

    def _random_crop(self, image, crop_size):
        h, w, _ = image.shape

        # Deicde top and left bitween 0 to (400-crop_size)
        top = np.random.randint(0, h - crop_size)
        left = np.random.randint(0, w - crop_size)

        # Decide bottom and right
        bottom = top + crop_size
        right = left + crop_size

        # Crop image using top,bottom,left,right
        image = image[top:bottom, left:right, :]
        return image

    def _scale_augmentation(self, image, crop_size, scale_range=(444, 600)):
        scale_size = np.random.randint(*scale_range)
        image = imresize(image, (scale_size, scale_size))
        image = self._random_crop(image, crop_size)
        return image


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
        flag = False
        while True:
            if len(self.images):
                inputs = np.asarray(self.images)
                targets = np.asarray(self.labels)
                self._reset()

                yield inputs, targets

            df = df.sample(frac=1).reset_index(drop=True)
            for i, row in df.iterrows():
                img = cv2.imread(dir_path + row.file_name)[:, :, ::-1]
                img = cv2.resize(img, (image_size, image_size))
                array_img = img_to_array(img)
                if self.augment:
                    array_img = self.datagen.random_transform(array_img)
                    if self.scale_augmentation:
                        array_img = self._scale_augmentation(array_img, image_size, scale_range=(444, 600))
                    if self.random_crop:
                        array_img = self._random_crop(array_img, image_size)
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
                    self._reset()
                    # mixup
                    if self.augment and self.mixup:
                        inputs, targets = self._mixup(inputs, targets)

                    yield inputs, targets

    def flow_from_dataframe_prediction(self, df, batch_size, image_size, dir_path):
        for i, row in df.iterrows():
            img = cv2.imread(dir_path + row.file_name)[:, :, ::-1]
            img = cv2.resize(img, (image_size, image_size))
            array_img = img_to_array(img)
            if self.augment:
                array_img = self.datagen.random_transform(array_img)
                if self.scale_augmentation:
                    array_img = self._scale_augmentation(array_img, image_size, scale_range=(444, 600))
                if self.random_crop:
                    array_img = self._random_crop(array_img, image_size)
                if self.random_erasing:
                    array_img = self._random_erasing(array_img)
            array_img = array_img/255
            array_img = np.expand_dims(array_img, axis=0)

            yield array_img

    def flow_from_list_prediction(self, lists, batch_size, image_size, dir_path):
        for file_name in lists:
            img = cv2.imread(dir_path + file_name)[:, :, ::-1]
            img = cv2.resize(img, (image_size, image_size))
            array_img = img_to_array(img)
            if self.augment:
                array_img = self.datagen.random_transform(array_img)
                if self.scale_augmentation:
                    array_img = self._scale_augmentation(array_img, image_size, scale_range=(444, 600))
                if self.random_crop:
                    array_img = self._random_crop(array_img, image_size)
                if self.random_erasing:
                    array_img = self._random_erasing(array_img)
            array_img = array_img/255
            array_img = np.expand_dims(array_img, axis=0)

            yield array_img
