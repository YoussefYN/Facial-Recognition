import random
from glob import glob
from random import randint

import numpy as np
import tensorflow as tf

from data_loader import Loader


class SiameseNet:

    def __init__(self, data_path, lr, seed=None):
        if seed is not None:
            self._determinize_model(seed)
        self._assemble_graph(lr)
        self._open_session()
        self.data_path = data_path + '/'
        self.images_mat = [glob(x + "/*") for x in glob(data_path + "/*")]
        self.idxs = [0] * len(self.images_mat)
        self.loader = Loader()

    @staticmethod
    def _model(input_):
        layer1 = tf.layers.dense(inputs=input_, units=512, activation='sigmoid')
        layer2 = tf.layers.dense(inputs=layer1, units=256, activation='sigmoid')
        layer3 = tf.nn.l2_normalize(x=layer2)
        layer4 = tf.layers.dense(inputs=layer3, units=128, activation='tanh')
        layer5 = tf.nn.l2_normalize(x=layer4)
        return layer5

    def _assemble_graph(self, lr=0.0001):
        self.input_anc = tf.placeholder(tf.float32, [None, 2048])
        self.input_pos = tf.placeholder(tf.float32, [None, 2048])
        self.input_neg = tf.placeholder(tf.float32, [None, 2048])
        self.alpha = tf.placeholder(tf.float32, ())
        with tf.variable_scope('siamese', reuse=tf.AUTO_REUSE):
            output_anc = self._model(self.input_anc)
            output_pos = self._model(self.input_pos)
            output_neg = self._model(self.input_neg)
        self.term1 = tf.square(tf.norm(output_anc - output_pos, axis=1))
        self.term2 = tf.square(tf.norm(output_anc - output_neg, axis=1))
        self.loss = tf.reduce_mean(tf.nn.relu(self.term1 + self.alpha - self.term2))
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

    def train(self, epochs=10, minibatch_size=256, alpha=0.2, thresh=-0.8):
        self.sess.run(tf.global_variables_initializer())
        import pandas as pd
        test_data = pd.read_csv('data/test_set.csv')
        for epoch in range(epochs):
            anc_batches, pos_batches, neg_batches = self._get_next_batch(minibatch_size)
            for anc_batch, pos_batch, neg_batch \
                    in zip(anc_batches, pos_batches, neg_batches):
                loss, _ = self.sess.run([self.loss, self.train_op],
                                        feed_dict={self.input_anc: anc_batch,
                                                   self.input_pos: pos_batch,
                                                   self.input_neg: neg_batch,
                                                   self.alpha: alpha})
            test_anc_batch = self.loader.load_preprocess_images(
                test_data['Anchor'].values, self.data_path)
            test_pos_batch = self.loader.load_preprocess_images(
                test_data['Positive'].values, self.data_path)
            test_neg_batch = self.loader.load_preprocess_images(
                test_data['Negative'].values, self.data_path)
            accuracy, test_loss = self._evaluate(test_anc_batch,
                                                 test_pos_batch,
                                                 test_neg_batch,
                                                 thresh=thresh,
                                                 alpha=alpha)

            print("Epoch %d, test acc %.4f, test batch loss %.4f" % (epoch, accuracy, test_loss))
        self.loader._save_cache()

    def _evaluate(self, anc_batch, pos_batch, neg_batch, thresh, alpha):
        same, different, loss = self.sess.run([self.term1, self.term2, self.loss],
                                              feed_dict={self.input_anc: anc_batch,
                                                         self.input_pos: pos_batch,
                                                         self.input_neg: neg_batch,
                                                         self.alpha: alpha})
        same_acc = np.where(same <= -1 * thresh, 1, 0)
        different_acc = np.where(different <= -1 * thresh, 0, 1)
        acc = np.append(same_acc, different_acc)
        return acc.mean(), loss

    def _get_next_batch(self, minibatch_sz):
        dirs_len = len(self.images_mat)
        anc_batch = []
        pos_batch = []
        neg_batch = []
        for i in range(dirs_len):
            images = self.images_mat[i]
            length = len(images)
            if length < 2:
                continue
            anc_batch.append(images[self.idxs[i]])
            pos_batch.append(images[(self.idxs[i] + 1) % length])
            self.idxs[i] = (self.idxs[i] + 1) % length
            rand_class = i
            while (len(self.images_mat[rand_class]) == 0) or (rand_class == i):
                rand_class = randint(0, dirs_len - 1)
            neg_batch.append(self.images_mat[rand_class][self.idxs[rand_class]])

        def divide(x):
            len_x = len(x)
            return [x[j: min(j + minibatch_sz, len_x)]
                    for j in range(0, len_x, minibatch_sz)]

        return divide(self.loader.load_preprocess_images(anc_batch)), \
               divide(self.loader.load_preprocess_images(pos_batch)), \
               divide(self.loader.load_preprocess_images(neg_batch))

    def _open_session(self):
        self.sess = tf.Session()

    @staticmethod
    def _determinize_model(seed):
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
