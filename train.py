# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

from SE_ResNeXt import SE_ResNeXt
from SE_Inception_resnet_v2 import SE_Inception_resnet_v2
from SE_Inception_v4 import SE_Inception_v4

from generator import DataGenerator


def Evaluate(sess, val_generator, test_iteration, epoch_learning_rate):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):

        test_batch_x, test_batch_y = next(val_generator)

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    # average loss
    test_loss /= test_iteration
    # average accuracy
    test_acc /= test_iteration

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


def main():
    # prepare dataframe
    df = pd.read_csv("../dataset/cookpad/train_master.tsv", delimiter="\t")
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask]
    val_df = df[~mask]
    category_df = pd.read_csv("../dataset/cookpad/master.tsv", delimiter="\t")

    # input params
    image_size = 224
    img_channels = 3

    # optimizer params
    weight_decay = 0.0005
    momentum = 0.9
    init_learning_rate = 0.1

    # network params
    # how many split
    cardinality = 8
    # res_block ! (split + transition)
    blocks = 3
    # out channel
    depth = 64
    reduction_ratio = 4

    # training params
    batch_size = 32
    iteration = int(len(train_df) / batch_size)
    test_iteration = int(len(val_df) / batch_size)
    total_epochs = 100
    nb_classes = len(category_df)


    # batch generators
    train_datagen = DataGenerator()
    train_generator = train_datagen.flow_from_dataframe(train_df, nb_classes, batch_size, image_size, augment=True)
    val_datagen = DataGenerator()
    val_generator = val_datagen.flow_from_dataframe(val_df, nb_classes, batch_size, image_size, augment=False)

    # placeholders
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
    label = tf.placeholder(tf.float32, shape=[None, class_num])

    # training flag: True or False
    training_flag = tf.placeholder(tf.bool)

    # learning rate
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    logits = SE_ResNeXt(x, training=training_flag).model
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    train_op = optimizer.minimize(loss + l2_loss * weight_decay)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./logs', sess.graph)

        epoch_learning_rate = init_learning_rate
        for epoch in range(1, total_epochs + 1):
            if epoch % 30 == 0 :
                epoch_learning_rate = epoch_learning_rate / 10

            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0

            for step in range(1, iteration + 1):

                batch_x, batch_y = next(train_generator)
                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size


            # average loss
            train_loss /= iteration
            # average accuracy
            train_acc /= iteration

            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

            test_acc, test_loss, test_summary = Evaluate(sess, val_generator, test_iteration, epoch_learning_rate)

            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()

            line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
            print(line)

            with open('logs.txt', 'a') as f:
                f.write(line)

            saver.save(sess=sess, save_path='./model/ResNeXt.ckpt')
