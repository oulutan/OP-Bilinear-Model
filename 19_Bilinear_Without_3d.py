"""Train and Eval the MNIST network.
This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See tensorflow/g3doc/how_tos/reading_data.md#reading-from-files
for context.
YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import tflearn
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import inception_v3_model.inception_v3 as inception_v3
from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.io as sio

import sys
from select import select

import ipdb

# SET THE GPU DEVICE
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# from tensorflow.examples.tutorials.mnist import mnist


# Basic model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_classes', 2, 'Number of classes for the last layer')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', None, 'Number of epochs to run trainer.')
flags.DEFINE_integer('max_steps', 40000, 'Total number of steps to train')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_string('train_dir', 'data/',
                    'Directory with the training data.')
flags.DEFINE_bool('isTest', False, 'Test Model')

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE_1 = 'avgOF_seismic_D2_1P_train_norm_edited_1.tfrecords'
TRAIN_FILE_2 = 'avgOF_seismic_D2_1P_train_norm_edited_2.tfrecords'
VALIDATION_FILE = 'avgOF_seismic_D2_1P_test_norm_edited.tfrecords'

IMAGE_FEATS_CKPT = 'inception_v3_model/inception_v3.ckpt'

IMAGE_CNN_CKPT = 'good_models/04_Images_Inception_2016-11-12_time_02-12_pooled.ckpt'
SEIS_CNN_CKPT = 'good_models/01_Seismic_2016-11-11_time_19-00_pooled.ckpt'

MODEL_CKPT = 'good_models/14_Smaller_Convs_2016-11-12_time_17-55.ckpt'
FINAL_CKPT = 'good_models/14_Smaller_Convs_2016-11-12_time_17-55.ckpt'

TRAIN_SIZE = 85964
VALID_SIZE = 32504
GPU_FRAC = 0.95

# IMAGE_SIZE = [299, 299]
IMAGE_SIZE = [150, 150]

NEARBY_TH = 15  # meters

fNAME = "19_Bilinear_Without_3d"
date = time.strftime("%Y-%m-%d_time_%H-%M")
NAME = fNAME + '_' + date

txt_logs_path = 'txt_logs/' + NAME + '.txt'
model_saver_path = 'model_weights/' + NAME + '.ckpt'


def remove_log():
    os.remove(txt_logs_path)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([100 * 100], tf.int64),
            'seismic': tf.FixedLenFeature([4096], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
            'ex_idx': tf.FixedLenFeature([], tf.int64),
            'cam_locs': tf.FixedLenFeature([4], tf.int64)

        })

    # TODO Dont forget to normalize and scale to [0, 1]

    image = tf.reshape(tf.cast(features['image'], tf.uint8), [100, 100, 1])
    label = tf.cast(features['label'], tf.int32)
    seismic = features['seismic']
    cam_loc = features['cam_locs']
    ex_idx = features['ex_idx']

    return label, image, seismic, cam_loc, ex_idx


def preprocess(labels, images, seismics):
    interpolated = tf.image.resize_bicubic(images, IMAGE_SIZE, name="interpolation")
    tiled = tf.tile(interpolated, [1, 1, 1, 3])
    p_images = (tiled / 256.0 - 0.5) * 2.0

    p_labels = tf.cast(tf.less_equal(labels, NEARBY_TH, name="label_quantization"), tf.uint8)
    onehot_labels = tf.one_hot(p_labels, 2, name="sparse_labels")

    p_seismics = tf.expand_dims(tf.expand_dims(seismics, 2), 3)
    return onehot_labels, p_images, p_seismics

def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    
    if train:
        filenames = [ os.path.join(FLAGS.train_dir,TRAIN_FILE_1), os.path.join(FLAGS.train_dir,TRAIN_FILE_2)]
    else:
        filenames = [os.path.join(FLAGS.train_dir,VALIDATION_FILE)]


    with tf.variable_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        label, image, seismic, cam_loc, ex_idx = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if train:
            labels, images, seismics, cam_locs, ex_idxs = tf.train.shuffle_batch(
                [label, image, seismic, cam_loc, ex_idx], batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
        else:
            labels, images, seismics, cam_locs, ex_idxs = tf.train.batch(
                [label, image, seismic, cam_loc, ex_idx], batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size)

    p_labels, p_images, p_seismics = preprocess(labels, images, seismics)
    return p_labels, p_images, p_seismics, cam_locs, ex_idxs


def image_features(images):
    with tf.variable_scope("Image_Feats"):
        #TODO add visual features
        with slim.arg_scope(inception_v3.inception_v3_arg_scope(weight_decay=0.0005)):
            with slim.arg_scope([slim.conv2d], trainable=True):
                # incp_feats, end_points = inception_v3.inception_v3_base(images, final_endpoint='Mixed_6e', scope='InceptionV3')
                incp_feats, end_points = inception_v3.inception_v3_base(images, final_endpoint='MaxPool_5a_3x3', scope='InceptionV3')
                # incp_feats, end_points = inception_v3.inception_v3_base(images, final_endpoint='Mixed_5d', scope='InceptionV3')

        # print(end_points)
        # vnet = slim.conv2d(images, 128, [3, 3])
        # vnet = slim.max_pool2d(vnet, [2, 2], stride = 2)
        # vnet = slim.conv2d(vnet, 256, [3, 3])
        # vnet = slim.max_pool2d(vnet, [2, 2], stride=2)
        # vnet = slim.conv2d(vnet, 256, [3, 3])
        # vnet = slim.max_pool2d(vnet, [2, 2], stride=2)
        # vis_feats = vnet

        # image_pooled = slim.max_pool2d(incp_feats, [4, 4], [2, 2])  # Bx7x7x192
        incp_feats_pooled = slim.conv2d(incp_feats, 20, [1, 1]
                                    , weights_initializer=tf.truncated_normal_initializer(stddev=0.1)
                                    , weights_regularizer=slim.l1_regularizer(0.00004))
        image_pooled = slim.max_pool2d(incp_feats_pooled, [4, 4], [2, 2])  # Bx7x7x192
        # ipdb.set_trace()
        vis_feats = image_pooled
    return vis_feats


def seismic_features(seismic):
    with tf.variable_scope("Seismic_Feats"):
        # seismic = tf.squeeze(seismic,[3])
        # snet = tflearn.conv_1d(seismic, 16, 10, activation='relu', weights_init='normal',)
        # # snet = tflearn.max_pool_1d(snet, 2)
        # snet = tflearn.conv_1d(snet, 16, 10, activation='relu', weights_init='normal')
        # snet = tflearn.max_pool_1d(snet, 2)
        # seismic_feats = snet

        snet = slim.conv2d(seismic, 32, [10, 1], weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(0.0001))
        snet = slim.conv2d(snet, 64, [10, 1], weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(0.0001))
        snet = slim.max_pool2d(snet, [10, 1], stride=[5, 1])

        snet = slim.conv2d(snet, 64, [5, 1], weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(0.0001))
        snet = slim.conv2d(snet, 64, [5, 1], weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(0.0001))
        snet = slim.max_pool2d(snet, [10, 1], stride=[5, 1])

        seismic_feats_n = slim.conv2d(snet, 20, [1, 1]
                                      , weights_initializer=tf.truncated_normal_initializer(stddev=0.1)
                                      , weights_regularizer=slim.l1_regularizer(0.0004))
        s_pooled = slim.max_pool2d(seismic_feats_n, [16, 1], [8, 1])

        seismic_feats = s_pooled
        # seismic_feats = snet
    return seismic_feats


def classification(final_feats, dropout_keep):
    with tf.variable_scope("classification"):
        # snet = tflearn.fully_connected(final_feats, 10, activation='relu')
        # snet = tflearn.dropout(snet, dropout_keep)
        # print(snet)
        # logits = tflearn.fully_connected(snet, FLAGS.num_classes, activation='linear')

        # fnet = final_feats
        #with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(scale=0.001 * 0.5)):
        #    fnet = slim.fully_connected(final_feats, 10, weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
        fnet = slim.fully_connected(final_feats, 32,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        weights_regularizer=slim.l2_regularizer(0.0001))
        fnet = slim.dropout(fnet, dropout_keep)
        logits = slim.fully_connected(fnet, FLAGS.num_classes, activation_fn=None
                                        , weights_initializer=tf.truncated_normal_initializer(stddev=0.1)
                                        , weights_regularizer=slim.l2_regularizer(0.0001))
    return logits


def inference(images, seismic, dropout_keep):
    with tf.variable_scope("inference"):
        # Image Model
        image_feats = image_features(images) # 7x7x20

        # Seismic Model
        seismic_feats = seismic_features(seismic) # 19x1x20

        # ipdb.set_trace()

        # Fusion
        with tf.variable_scope("Fusion"):


            F1 = tf.reshape(image_feats, [FLAGS.batch_size, -1, 1])

            F2 = tf.reshape(seismic_feats, [FLAGS.batch_size, 1, -1])

            i_shape = image_feats.get_shape()
            s_shape = seismic_feats.get_shape()
            output_shape = [FLAGS.batch_size, int(i_shape[1]), int(i_shape[2]), int(s_shape[1]),
                            int(i_shape[3] * s_shape[3])]  # Bx7x7x39x3072

            # F1_norm = slim.batch_norm
            # bil_feats = tf.batch_matmul(F1, F2, name="Tensor_Product")
            bil_feats = tf.matmul(F1, F2, name="Tensor_Product")
            # bil_feats = tf.batch_matmul(F1, F2, name="Tensor_Product")

            # ipdb.set_trace()

            spatio_temporal_feats = tf.reshape(bil_feats, output_shape)

            # ipdb.set_trace()



            # c3_1 = tflearn.conv_3d(spatio_temporal_feats, 128, 3, activation='relu', weights_init='normal')
            # c3_p = tflearn.max_pool_3d(c3_1, kernel_size=[1, 2, 2, 5, 1], strides=[1, 2, 2, 5, 1]) # 4x4x4x128

            # tf.add_to_collection('WtoRegu', c3_1.W)

            # ipdb.set_trace()

            # c3_2 = tflearn.conv_3d(c3_p, 64, 3, activation='relu', weights_init='normal')
            # c3_p2 = tflearn.max_pool_3d(c3_2, kernel_size=[1, 2, 2, 5, 1], strides=[1, 2, 2, 5, 1])

            # ipdb.set_trace()

        final_feats = tf.reshape(spatio_temporal_feats, [FLAGS.batch_size, -1])

        # ipdb.set_trace()

        # Classifier
        logits = classification(final_feats, dropout_keep)

    return logits


def loss_op(y_probs, labels):
    """Calculates the loss from the logits and the labels.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].
    Returns:
      loss: Loss tensor of type float.
    """

    # %%Get class weights

    # weights = np.zeros(labels_.shape)

    # for ii in range(0, labels_.size):
    # #     weights[ii] = train_labels.size/ (float(train_labels[train_labels == labels_[ii]].size) * labels_.size)
    #     weights[ii] = (labels_quant.size - float(labels_quant[labels_quant == labels_[ii]].size) ) / labels_quant.size

    # weights = [0.06561035, 0.93438965]
    # weights = [0.1373, 0.8627]

    # weights = [0.1, 0.9]
    weights = [1, 1]

    weights = np.expand_dims(weights, axis=1)
    # Labels are one-hot matrices, weights are Nclass x 1 vectors, matmul between will give a vector where each
    # sample has the corresponding class weight in its corresponding row
    with tf.name_scope("Class_Weights"):
        corresponding_weights = tf.matmul(tf.cast(labels, tf.float32), tf.cast(weights, tf.float32))

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(corresponding_weights * labels * tf.log(tf.clip_by_value(y_probs, 1e-10, 1e10)),
                           reduction_indices=[1]))
    with tf.name_scope("Regularizers"):
        # spec_regu = tf.reduce_sum(tf.abs(tf.get_collection('WtoRegu')[0]))
        spec_regu = 0

        # regularizers_fusion = tf.add_n(slim.losses.get_regularization_losses('inference/Fusion'))
        regularizers_inception = tf.add_n(slim.losses.get_regularization_losses('inference/Image_Feats'))
        regularizers_FC = tf.add_n(slim.losses.get_regularization_losses('inference/classification'))
        regularizers_SCNN = tf.add_n(slim.losses.get_regularization_losses('inference/Seismic_Feats'))

        
        regularizers = regularizers_inception + regularizers_FC + regularizers_SCNN + spec_regu * 1e-6 # + regularizers_fusion
        #tf.add_to_collection('Regularizers_List', [regularizers_fusion, regularizers_inception, regularizers_FC, spec_regu * 1e-6])

    with tf.name_scope("Total_Loss"):
        loss = cross_entropy + regularizers
    # tf.add_to_collection('Losses', [regularizers_inception, regularizers_FC, cross_entropy])
    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    total_loss = loss
    if batchnorm_updates:
        total_loss = control_flow_ops.with_dependencies(batchnorm_updates, total_loss)
    return total_loss, cross_entropy


def training(loss, learning_rate=1e-4):
    """Sets up the training Ops.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    with tf.name_scope("ADAM_optimizer"):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-1).minimize(loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    return train_step


def run_eval(sess, y_probs, loss, labels, dropout_keep, size):
    predictions = np.zeros((size,))
    true_labels = np.zeros((size,))
    total_loss = 0
    no_batches = int(np.ceil(size * 1.0 / FLAGS.batch_size))
    for ii in tqdm(range(no_batches)):
        probs, trues, loss_i = sess.run([y_probs, labels, loss], feed_dict={dropout_keep: 1.0})
        preds = np.argmax(probs, axis=1)
        t_labels = np.argmax(trues, axis=1)
        s_idx = ii * FLAGS.batch_size
        e_idx = (ii + 1) * FLAGS.batch_size
        e_queue = FLAGS.batch_size
        if e_idx > size: e_idx = size; e_queue = e_idx - s_idx
        predictions[s_idx:e_idx] = preds[0:e_queue]
        true_labels[s_idx:e_idx] = t_labels[0:e_queue]

        total_loss = total_loss + loss_i
    return predictions, true_labels, total_loss * 1.0 / no_batches

def run_eval_testing(sess, y_probs, loss, labels, idcs, dropout_keep, size):
    all_probs = np.zeros((size, FLAGS.num_classes))
    all_indices = np.zeros((size,))
    predictions = np.zeros((size,))
    true_labels = np.zeros((size,))
    total_loss = 0
    no_batches = int(np.ceil(size * 1.0 / FLAGS.batch_size))
    for ii in tqdm(range(no_batches)):
        probs, trues, loss_i, idx = sess.run([y_probs, labels, loss, idcs], feed_dict={dropout_keep: 1.0})
        preds = np.argmax(probs, axis=1)
        t_labels = np.argmax(trues, axis=1)
        s_idx = ii * FLAGS.batch_size
        e_idx = (ii + 1) * FLAGS.batch_size
        e_queue = FLAGS.batch_size
        if e_idx > size: e_idx = size; e_queue = e_idx - s_idx
        predictions[s_idx:e_idx] = preds[0:e_queue]
        true_labels[s_idx:e_idx] = t_labels[0:e_queue]
        all_probs[s_idx:e_idx, :] = probs[0:e_queue]
        all_indices[s_idx:e_idx] = idx[0:e_queue]

        total_loss = total_loss + loss_i
    return predictions, true_labels, total_loss * 1.0 / no_batches, all_probs, all_indices


def run_training():
    """Train Model for a number of steps."""
    print("Running %s" % NAME)
    print("Logging to %s" % txt_logs_path)

    keep_prob = 0.8

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        #### TRAINING PORTION ####

        # Input images and labels.

        labels_tr, images_tr, seismics_tr, cam_locs_tr, ex_idxs_tr = inputs(train=True, batch_size=FLAGS.batch_size,
                                                                            num_epochs=FLAGS.num_epochs)
        dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")

        # Build a Graph that computes predictions from the inference model.
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=True):
            logits_tr = inference(images_tr, seismics_tr, dropout_keep)

        print([v.name for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="inference/Image_Feats/InceptionV3")][0])
        print('Printing the type of get_collection')
        print(type(tf.get_collection(tf.GraphKeys.VARIABLES, scope="inference/Image_Feats/InceptionV3")))

        # ipdb.set_trace()

        y_probs_tr = tf.nn.softmax(logits_tr, name="Softmax_Layer")

        # Add to the Graph the loss calculation.
        loss_tr, xent_tr = loss_op(y_probs_tr, labels_tr)
        # tf.scalar_summary("Training_Loss", loss_tr)

        # Add to the Graph operations that train the model.
        train_op = training(loss_tr, FLAGS.learning_rate)

        # The op for initializing the variables.
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())  # local is needed to initialize number of epochs in queue

        #### VALIDATION PORTION ####

        tf.get_variable_scope().reuse_variables()
        # print(v for v in slim.variables.get_variables())

        # Input images and labels.
        labels_tst, images_tst, seismics_tst, cam_locs_tst, ex_idxs_tst = inputs(train=False,
                                                                                 batch_size=FLAGS.batch_size,
                                                                                 num_epochs=FLAGS.num_epochs)

        # Build a Graph that computes predictions from the inference model.
        # with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
        with slim.arg_scope([slim.dropout], is_training=False):
            with slim.arg_scope([slim.batch_norm], is_training=False):
                logits_tst = inference(images_tst, seismics_tst, dropout_keep)

        y_probs_tst = tf.nn.softmax(logits_tst, name="Softmax_Layer")

        # Add to the Graph the loss calculation.
        loss_tst, xent_tst = loss_op(y_probs_tst, labels_tst)
        # tf.scalar_summary("Testing_Loss", loss_tst)

        ## Savers
        # Inception Save, dont need it anymore
        # image_feats_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="inference/Image_Feats/InceptionV3")
        # image_feats_dict = {v.op.name.replace("inference/Image_Feats/", ""): v for v in image_feats_vars}
        # image_feats_saver = tf.train.Saver(image_feats_dict)

        image_cnn_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="inference/Image_Feats/InceptionV3")
        image_cnn_dict = {v.op.name: v for v in image_cnn_vars if not 'Adam' in v.name}
        image_cnn_saver = tf.train.Saver(image_cnn_dict)

        seis_cnn_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="inference/Seismic_Feats")
        seis_cnn_dict = {v.op.name: v for v in seis_cnn_vars if not 'Adam' in v.name}
        seis_cnn_saver = tf.train.Saver(seis_cnn_dict)

        model_saver = tf.train.Saver()

        wo_optimizer_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if not 'Adam' in var.name]
        wo_optimizer_dict = {v.op.name: v for v in wo_optimizer_vars}
        wo_optimizer_saver = tf.train.Saver(wo_optimizer_dict)
        # ipdb.set_trace()
        #### Start Session ####

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_FRAC)
        # config = tf.ConfigProto(gpu_options=gpu_options)
        # Create a session for running operations in the Graph.
        # sess = tf.Session(config)
        sess = tf.Session()

        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)

        # Load pretrained feature models

        # ipdb.set_trace()
        # Inception
        # image_feats_saver.restore(sess, IMAGE_FEATS_CKPT)

        # Pretrained Models
        # image_cnn_saver.restore(sess, IMAGE_CNN_CKPT)
        # seis_cnn_saver.restore(sess, SEIS_CNN_CKPT)
        image_cnn_saver.restore(sess, MODEL_CKPT)
        seis_cnn_saver.restore(sess, MODEL_CKPT)

        # Same model retrained
        # model_saver.restore(sess, MODEL_CKPT)

        if (FLAGS.isTest):
            #model_saver.restore(sess, FINAL_CKPT)
            wo_optimizer_saver.restore(sess, FINAL_CKPT)

        # ipdb.set_trace()

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # ipdb.set_trace()
        if not FLAGS.isTest:
            # Start the text logger
            f = open(txt_logs_path, 'w')

            f.write('Logging for file %s \n' % NAME)

            avg_loss = 0
            avg_xent = 0

            for step in xrange(1, FLAGS.max_steps + 1):

                start_time = time.time()
                _, loss_value, xent_value = sess.run([train_op, loss_tr, xent_tr], feed_dict={dropout_keep: keep_prob})
                duration = time.time() - start_time
                avg_loss = avg_loss + loss_value
                avg_xent = avg_xent + xent_value

                step_size = 100
                if step % step_size == 0:
                    examples_per_sec = FLAGS.batch_size / float(duration)
                    format_str = ('%s.py %s: step %d, (xent+regu)/batch = %.5f, xent/batch = %.5f (%.3f sec/batch)')
                    p_str = format_str % (fNAME[0:2], time.strftime("%H:%M"), step, avg_loss * 1.0 / step_size, avg_xent * 1.0 / step_size, duration)
                    print(p_str)
                    f.write(p_str + '\n')  # Logging to file
                    avg_loss = 0
                    avg_xent = 0

                #if step % 1000 == 0:
                if step % (int(np.ceil(TRAIN_SIZE * 1.0 / FLAGS.batch_size))) == 0:
                    predictions_tst, true_labels_tst, total_loss_tst = run_eval(sess, y_probs_tst, xent_tst, labels_tst,
                                                                                dropout_keep, VALID_SIZE)

                    target_names = ['no_person', 'person']
                    p_str = ("********Classification Report for Testing Set********\n" +
                             "LOSS FOR TESTING : %.5f \n" % (total_loss_tst) +
                             classification_report(true_labels_tst, predictions_tst, target_names=target_names))

                    print(p_str)
                    f.write(p_str + '\n')

                    model_saver.save(sess, model_saver_path)
                    print('Checkpoint Saved')

                #if step % 10000 == 0:
                if step % (int(np.ceil(TRAIN_SIZE * 1.0 / FLAGS.batch_size))*5) == 0:
                    predictions_tr, true_labels_tr, total_loss_tr = run_eval(sess, y_probs_tr, xent_tr, labels_tr,
                                                                             dropout_keep, TRAIN_SIZE)
                    p_str = ('********Classification Report for Training Set********\n' +
                             'LOSS FOR TRAINING : %.5f \n' % (total_loss_tr) +
                             classification_report(true_labels_tr, predictions_tr, target_names=target_names))

                    print(p_str)
                    f.write(p_str + '\n')
            # After the loop is done run results one more time 

            predictions_tst, true_labels_tst, total_loss_tst = run_eval(sess, y_probs_tst, xent_tst, labels_tst,
                                                                        dropout_keep, VALID_SIZE)

            target_names = ['no_person', 'person']
            p_str = ("********Classification Report for Testing Set********\n" +
                     "LOSS FOR TESTING : %.5f \n" % (total_loss_tst) +
                     classification_report(true_labels_tst, predictions_tst, target_names=target_names))

            print(p_str)
            f.write(p_str + '\n')

            model_saver.save(sess, model_saver_path)
            print('Checkpoint Saved')

            predictions_tr, true_labels_tr, total_loss_tr = run_eval(sess, y_probs_tr, xent_tr, labels_tr,
                                                                     dropout_keep, TRAIN_SIZE)
            p_str = ('********Classification Report for Training Set********\n' +
                     'LOSS FOR TRAINING : %.5f \n' % (total_loss_tr) +
                     classification_report(true_labels_tr, predictions_tr, target_names=target_names))

            print(p_str)
            f.write(p_str + '\n')

        else:
            predictions_tst, true_labels_tst, total_loss_tst, probs_tst, indices_tst = run_eval_testing(sess,
                                                                                                        y_probs_tst,
                                                                                                        xent_tst,
                                                                                                        labels_tst,
                                                                                                        ex_idxs_tst,
                                                                                                        dropout_keep,
                                                                                                        VALID_SIZE)

            target_names = ['no_person', 'person']
            p_str = ("********Classification Report for Testing Set********\n" +
                     "LOSS FOR TESTING : %.5f \n" % (total_loss_tst) +
                     classification_report(true_labels_tst, predictions_tst, target_names=target_names))

            print(p_str)

            cmtx_tst = confusion_matrix(true_labels_tst, predictions_tst)
            cmtx_tst_normalized = cmtx_tst.astype('float') / cmtx_tst.sum(axis=1)[:, np.newaxis]
            print(cmtx_tst)
            print(cmtx_tst_normalized)

            precision, recall, thresholds = precision_recall_curve(true_labels_tst, probs_tst[:, 1])
            plt.plot(precision, recall)

            filelabel = 'bilinear'
            plt.savefig('results/prec_recall_' + filelabel + '.png')

            ids = indices_tst[true_labels_tst==1][np.argsort(probs_tst[true_labels_tst==1][:,1])]
            probs = probs_tst[true_labels_tst==1][np.argsort(probs_tst[true_labels_tst==1][:,1])]

            matdict = {'ids_'+filelabel:ids, 'probs_'+filelabel:probs, 'precision_'+filelabel:precision, 'recall_'+filelabel:recall}

            sio.savemat('results/'+filelabel+'_res.mat', matdict)

            ipdb.set_trace()
            # indices_tst[true_labels_tst==1][np.argsort(probs_tst[true_labels_tst==1][:,1])[0:15]] # get worst performer IDs
            # indices_tst[true_labels_tst==1][np.argsort(probs_tst[true_labels_tst==1][:,0])[0:15]] # get best performer IDs
            # np.save('worst_to_best_prob_ids_bilinear.npy', indices_tst[true_labels_tst==1][np.argsort(probs_tst[true_labels_tst==1][:,1])])

            # probs_tst[true_labels_tst == 1, :]
            # indices_tst[true_labels_tst == 1][np.greater(probs_tst[true_labels_tst == 1, 0], 0.9)]


            predictions_tr, true_labels_tr, total_loss_tr, probs_tr, indices_tr = run_eval_testing(sess, y_probs_tr,
                                                                                                   xent_tr, labels_tr,
                                                                                                   ex_idxs_tr,
                                                                                                   dropout_keep,
                                                                                                   TRAIN_SIZE)

            p_str = ('********Classification Report for Training Set********\n' +
                     'LOSS FOR TRAINING : %.5f \n' % (total_loss_tr) +
                     classification_report(true_labels_tr, predictions_tr, target_names=target_names))

            print(p_str)

            cmtx_tr = confusion_matrix(true_labels_tr, predictions_tr)
            cmtx_tr_normalized = cmtx_tr.astype('float') / cmtx_tr.sum(axis=1)[:, np.newaxis]
            print(cmtx_tr)
            print(cmtx_tr_normalized)

            # try:
        # step = 0
        #   while not coord.should_stop():
        #     start_time = time.time()
        #
        #     # Run one step of the model.  The return values are
        #     # the activations from the `train_op` (which is
        #     # discarded) and the `loss` op.  To inspect the values
        #     # of your ops or variables, you may include them in
        #     # the list passed to sess.run() and the value tensors
        #     # will be returned in the tuple from the call.
        #     _, loss_value, labels_batch = sess.run([train_op, loss, labels])
        #
        #     duration = time.time() - start_time
        #
        #     # Print an overview fairly often.
        #     if step % 100 == 0:
        #       print('Step %d: loss = %.2f (%.3f sec), with batch %d' % (step, loss_value,
        #                                                  duration, labels_batch.shape[0]))
        #     step += 1
        # except tf.errors.OutOfRangeError:
        #   print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        # finally:
        #   # When done, ask the threads to stop.
        #   coord.request_stop()

        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
    # main()
