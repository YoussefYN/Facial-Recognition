import random

import numpy as np
import tensorflow as tf

from data_loader import Loader


class SiameseNet:

    def __init__(self, data_path, inception_path, lr, seed=None):
        if seed is not None:
            self._determinize_model(seed)
        self._assemble_graph(lr)
        self._open_session()
        self._loader = Loader(data_path, inception_path)

    @staticmethod
    def _model(input_):
        layer1 = tf.layers.dense(inputs=input_, units=512, activation='sigmoid')
        layer2 = tf.layers.dense(inputs=layer1, units=256, activation='sigmoid')
        layer3 = tf.nn.l2_normalize(x=layer2)
        layer4 = tf.layers.dense(inputs=layer3, units=128, activation='tanh')
        layer5 = tf.nn.l2_normalize(x=layer4)
        return layer5

    def _assemble_graph(self, lr=0.0001):
        self._input_anc = tf.placeholder(tf.float32, [None, 2048])
        self._input_pos = tf.placeholder(tf.float32, [None, 2048])
        self._input_neg = tf.placeholder(tf.float32, [None, 2048])
        self._alpha = tf.placeholder(tf.float32, ())
        with tf.variable_scope('siamese', reuse=tf.AUTO_REUSE):
            output_anc = self._model(self._input_anc)
            output_pos = self._model(self._input_pos)
            output_neg = self._model(self._input_neg)
        self._term1 = tf.square(tf.norm(output_anc - output_pos, axis=1))
        self._term2 = tf.square(tf.norm(output_anc - output_neg, axis=1))
        self._loss = tf.reduce_mean(tf.nn.relu(self._term1 + self._alpha - self._term2))
        self._train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._loss)

    def train(self, epochs=10, minibatch_size=256, alpha=0.2, thresh=-0.8):
        self._sess.run(tf.global_variables_initializer())
        test_anc_batch, test_pos_batch, test_neg_batch = self._loader.get_test_batches()

        for epoch in range(epochs):
            anc_batches, pos_batches, neg_batches = self._loader.get_next_batch(minibatch_size)
            for anc_batch, pos_batch, neg_batch \
                    in zip(anc_batches, pos_batches, neg_batches):
                loss, _ = self._sess.run([self._loss, self._train_op],
                                         feed_dict={self._input_anc: anc_batch,
                                                   self._input_pos: pos_batch,
                                                   self._input_neg: neg_batch,
                                                   self._alpha: alpha})

            accuracy, test_loss = self._evaluate(test_anc_batch,
                                                 test_pos_batch,
                                                 test_neg_batch,
                                                 thresh=thresh,
                                                 alpha=alpha)

            print("Epoch %d, test acc %.4f, test batch loss %.4f" % (epoch, accuracy, test_loss))
        self._loader._save_cache()

    def _evaluate(self, anc_batch, pos_batch, neg_batch, thresh, alpha):
        same, different, loss = self._sess.run([self._term1, self._term2, self._loss],
                                               feed_dict={self._input_anc: anc_batch,
                                                         self._input_pos: pos_batch,
                                                         self._input_neg: neg_batch,
                                                         self._alpha: alpha})
        same_acc = np.where(same <= -1 * thresh, 1, 0)
        different_acc = np.where(different <= -1 * thresh, 0, 1)
        acc = np.append(same_acc, different_acc)
        return acc.mean(), loss

    def _open_session(self):
        self._sess = tf.Session()

    @staticmethod
    def _determinize_model(seed):
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
