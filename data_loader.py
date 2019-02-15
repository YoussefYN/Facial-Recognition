import os
import pickle
from random import randint

import cv2
import tensorflow as tf
from pandas import read_csv


class Loader:

    def __init__(self, data_path, path, im_size=299):
        self.model_path = path
        self.im_size = im_size
        self._assemble_cnn_graph()
        self._open_session()
        self.data_path = data_path
        self.images_mat = [[os.path.join(x, y) for y in os.listdir(os.path.join(self.data_path, x))]
                           for x in os.listdir(self.data_path)]
        self.idxs = [0] * len(self.images_mat)
        try:
            self.cache = pickle.load(open("img_cach_file.pkl", "rb"))
            print("Re-read Cache")
        except (OSError, IOError):
            self.cache = dict()

    def _assemble_cnn_graph(self):
        with tf.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            self.input_ = tf.get_default_graph().get_tensor_by_name("Model_Input:0")
            self.cnn_embedding_ = tf.get_default_graph().get_tensor_by_name("Model_Output:0")

    def _load_image(self, path):
        image = cv2.resize(cv2.imread(os.path.join(self.data_path, path)), (self.im_size, self.im_size))
        embedding = self.sess.run(self.cnn_embedding_, feed_dict={self.input_: [image]})[0]
        self.cache[path] = embedding
        return embedding

    def _load_preprocess_images(self, batch):
        return [self.cache[p] if p in self.cache else self._load_image(p)
                for p in batch]

    def _open_session(self):
        self.sess = tf.Session()

    def _save_cache(self):
        file = open("img_cach_file.pkl", "wb")
        pickle.dump(self.cache, file)
        file.close()

    def get_next_batch(self, minibatch_sz):
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

        return divide(self._load_preprocess_images(anc_batch)), \
               divide(self._load_preprocess_images(pos_batch)), \
               divide(self._load_preprocess_images(neg_batch))

    def get_test_batches(self):
        test_data = read_csv('test_set.csv')
        test_anc_batch = self._load_preprocess_images(
            test_data['Anchor'].values)
        test_pos_batch = self._load_preprocess_images(
            test_data['Positive'].values)
        test_neg_batch = self._load_preprocess_images(
            test_data['Negative'].values)
        return test_anc_batch, test_pos_batch, test_neg_batch
