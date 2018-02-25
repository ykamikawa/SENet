# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import sys
from datetime import datetime

from SE_ResNeXt import SE_ResNeXt
from SE_Inception_resnet_v2 import SE_Inception_resnet_v2
from SE_Inception_v4 import SE_Inception_v4

from generator import DataGenerator


def Train(args):
    # prepare dataframe
    df = pd.read_csv(args.data_list, delimiter="\t")
    train_df, val_df = train_test_split(df, test_size=0.2)
    category_df = pd.read_csv(args.category_list, delimiter="\t")

    # input params
    image_size = args.input_size
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
    batch_size = args.batch_size
    iteration = int(len(train_df) / batch_size)
    test_iteration = int(len(val_df) / batch_size)
    total_epochs = args.epochs
    nb_classes = len(category_df) + 1


    # batch generators
    train_datagen = DataGenerator()
    train_generator = train_datagen.flow_from_dataframe(
            train_df,
            nb_classes,
            batch_size,
            image_size,
            augment=True)
    val_datagen = DataGenerator()
    val_generator = val_datagen.flow_from_dataframe(
            val_df,
            nb_classes,
            batch_size,
            image_size,
            augment=False)

    # placeholders
    x = tf.placeholder(
            tf.float32,
            shape=[None, image_size, image_size, img_channels])
    label = tf.placeholder(
            tf.float32,
            shape=[None, nb_classes])

    # training flag: True or False
    training_flag = tf.placeholder(tf.bool)

    # learning rate
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # build model
    if args.architecture == "SE_ResNeXt":
        logits = SE_ResNeXt(
                x,
                nb_classes=nb_classes,
                training=training_flag,
                blocks=blocks,
                cardinality=cardinality,
                depth=depth,
                ratio=reduction_ratio).model
    elif args.architecture == "SE_Inception_v4":
        logits = SE_Inception_v4(
                x,
                nb_classes=nb_classes,
                training=training_flag,
                ratio=reduction_ratio).model
    elif args.architecture == "SE_Inception_resnet_v2":
        logits = SE_Inception_resnet_v2(
                x,
                nb_classes,
                training=training_flag,
                ratio=reduction_ratio).model

    # loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    # optimizer
    if args.optimizer == "momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    elif args.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif args.optimizer == "SGD":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif args.optimizer == "SGD":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss + l2_loss * weight_decay)

    # caluc accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # trained model saver
    saver = tf.train.Saver(tf.global_variables())

    # start session
    with tf.Session() as sess:
        # get time
        start_time = str(datetime.now())

        # ckpt config
        ckpt_counter = len([ckpt_dir for ckpt_dir in os.listdir("./model") if args.architecture in ckpt_dir])
        save_dir = "./model/" + args.architecture + "_" + str(ckpt_counter + 1)
        save_path = save_dir + "/" + args.architecture + start_time +".ckpt"
        ckpt = tf.train.get_checkpoint_state("./model/" + args.architecture + "_" + str(ckpt_counter))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # log config
        log_counter = len([log_dir for log_dir in os.listdir("./logs") if args.architecture in log_dir])
        log_dir = "./logs/" + args.architecture + "_" + str(log_counter + 1)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # training
        epoch_learning_rate = init_learning_rate
        for epoch in range(1, total_epochs + 1):
            if epoch % 30 == 0 :
                epoch_learning_rate = epoch_learning_rate / 10

            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0

            # get epoch start time
            start_epoch = datetime.now()

            for step in range(1, iteration + 1):
                batch_x, batch_y = next(train_generator)
                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True}

                _, batch_loss = sess.run([train_op, loss], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                # get step elapsed time
                elapsed_step = datetime.now() - start_epoch

                sys.stdout.write("\r epoch: {} step: {}/{} loss: {} acc: {} elapsed: {}".format(epoch, step, iteration, batch_loss, batch_acc, str(elapsed_step)))
                sys.stdout.flush()

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

            # get elapsed time
            elapsed_epoch = datetime.now() - start_epoch

            # average loss
            train_loss /= iteration
            # average accuracy
            train_acc /= iteration

            # write train summary
            train_summary = tf.Summary(
                    value=[
                        tf.Summary.Value(
                            tag='train_loss',
                            simple_value=train_loss),
                        tf.Summary.Value(
                            tag='train_accuracy',
                            simple_value=train_acc)])

            # validation
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
                    training_flag: False}

                loss_, acc_ = sess.run([loss, accuracy], feed_dict=test_feed_dict)
                test_loss += loss_
                test_acc += acc_

            # average loss
            test_loss /= test_iteration
            # average accuracy
            test_acc /= test_iteration

            # write validation summary
            test_summary = tf.Summary(
                    value=[
                        tf.Summary.Value(
                            tag='test_loss',
                            simple_value=test_loss),
                        tf.Summary.Value(
                            tag='test_accuracy',
                            simple_value=test_acc)])

            # write summary
            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()

            # stdout line
            line = "\n epoch:{0}/{1} train_loss:{2} train_acc:{3} test_loss:{4} test_acc:{5} time:{6} elapsed:{7}".format(epoch, total_epochs, train_loss, train_acc, test_loss, test_acc, str(datetime.now()), str(elapsed_epoch))
            print(line)

            # logs text
            logs_text = "./log_txt/" + args.architecture + "_logs.txt"
            with open(logs_text, 'a') as f:
                f.write(line)

            # save ckpt
            saver.save(sess=sess, save_path=save_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='SENet')
    argparser.add_argument(
        '-d',
        '--data_list',
        type=str,
        default="../dataset/cookpad/train_master.tsv",
        help='path to data list')
    argparser.add_argument(
        "-c",
        '--category_list',
        type=str,
        default="../dataset/cookpad/master.tsv",
        help='path to data list')
    argparser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=100,
        help='number of epochs')
    argparser.add_argument(
        "-s",
        '--input_size',
        type=int,
        default=224,
        help='input size')
    argparser.add_argument(
        "-b",
        '--batch_size',
        default=4,
        type=int,
        help='batch size')
    argparser.add_argument(
        '-a',
        '--architecture',
        type=str,
        default="SE_ResNeXt",
        help='model architecture')
    argparser.add_argument(
        '--optimizer',
        type=str,
        default="adam",
        help='train optimizer')
    argparser.add_argument(
        "-g",
        '--gpu_id',
        default="0",
        type=str,
        help='gpu id')
    args = argparser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    Train(args)
